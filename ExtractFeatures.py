# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 23:04:49 2016

@author: liam
"""

import numpy as np
from wavio2 import readwav
from scipy.fftpack import fft as scifft
from id3reader import Reader

class WavFeatureExtractor():
    def __init__(self, downsample, desired_X_time_dim, 
                 track_fnames, tracks_path, rng=0,
                 fft_sample_length=None, fft_step_size=None,
                 desired_X_raw_seconds=None, complete_track_mode=False):
        
        self.downsample = downsample
        self.rng = rng
        self.desired_rate = 44100
        self.track_fnames = track_fnames
        self.desired_X_time_dim = desired_X_time_dim
        self.fft_sample_length = fft_sample_length
        self.fft_step_size = fft_step_size
        self.tracks_path = self.clean_tracks_path(tracks_path)
        self.curr_track_idx = -1
        #assert self.fft_sample_length % self.fft_step_size == 0
        if fft_step_size is not None:
            clip_seconds = ((self.desired_X_time_dim - 1) * \
                             self.fft_step_size + self.fft_sample_length) * \
                             (1. / self.desired_rate)
                             
            print 'each clip will be', clip_seconds, 'seconds long'
            self.fps = self.desired_X_time_dim / clip_seconds
            print 'resolution is', self.fps, 'frames per second'
        
        # For raw .wav mode:
        self.desired_X_raw_seconds = desired_X_raw_seconds
        self.complete_track_mode = complete_track_mode
        if self.complete_track_mode:
            self.desired_X_raw_seconds_orig = self.desired_X_raw_seconds
            self.y_fps_orig = self.desired_X_time_dim / \
                              self.desired_X_raw_seconds
    
    def clean_tracks_path(self, tracks_path):
        
        # Remove leading and trailing slashes from relative .wav filepath
        if tracks_path.endswith('/'):
            tracks_path = tracks_path[:-1]
        if tracks_path.startswith('/'):
            tracks_path = tracks_path[1:]
        return tracks_path
    
    def preprocess_wav(self, wav, cut_silence=True):
        
        # Stereo to mono
        wav = np.mean(wav, axis=1)
        
        # Normalise
        wav = wav / np.max(wav)
        
        # > 0.03 to cut off any silence at the beginning of the song
        if cut_silence:
            wav = wav[np.where(np.abs(wav) > 0.03)[0][0]:]
            
        return wav

    def get_spectogram(self, wav):
        
        spectra = []
        
        # Adding some noise, mainly as an alternative to log(1 + wav)
        wav += np.abs(np.random.rand(len(wav)) * 0.0001)
        
        # We're going to chop the wav into sections of fft_sample_length,
        # possibly overlapping if the fft_step_size is smaller than this
        spectra = np.array(
            [get_fft(wav[i:(i + self.fft_sample_length)]) for i in \
             range(0, len(wav) - self.fft_sample_length, self.fft_step_size)])
             
        # Normalise and take logs
        spectra = np.log(spectra / np.max(spectra))
        
        # Check the spectra is not garbage
        if (np.sum(np.isnan(spectra)) > 0 or np.sum(np.abs(spectra)) < 100.):
            return None
        
        # Downsample the frequency bins if we so desire
        spectra = np.mean(spectra.reshape(
            spectra.shape[0], -1 , self.downsample, order='F'), axis=1)

        # Normalise this so it's between 0 and 1
        return normalise(spectra).astype(np.float32)         
    
    def get_wav(self, track_full_path, seconds_cutoff=0):
        
        if track_full_path.endswith('.mp3'):
            # Convert the mp3 to a temporary wav file        
            wav_path = '/tmp/tmp-bpm.wav'
            convert_an_mp3_to_wav(track_full_path, wav_path)
            cut_silence = True
        
        else:
            wav_path = track_full_path
            cut_silence = False
            
        # Read in the wav
        rate, sampwidth, wav = readwav(wav_path)
        
        # Preprocess it (cut off silence and normalise)
        wav = self.preprocess_wav(wav, cut_silence=cut_silence)
        
        # Make sure it's the right sampling rate
        if rate != self.desired_rate:
            return None
        
        if seconds_cutoff:
            wav_complete_seconds = int(np.floor(len(wav) / rate))
            wav = wav[:min(seconds_cutoff, wav_complete_seconds) * rate]
            
        return wav
    
    def append_wav_to_Xy(self, wav):
        
        # Get the spectogram for the current wav
        X_curr = self.get_spectogram(wav)
        
        if X_curr is not None:
            
            # Get the number of seconds of audio represented by X_curr
            track_seconds = ((X_curr.shape[0] - 1) * self.fft_step_size + \
                             self.fft_sample_length) * (1. / self.desired_rate)
            
            # Get a target vector of the same length representing the same
            # number of seconds
            y_curr = get_target_vector(self.bpm, track_seconds, 
                                       resolution=X_curr.shape[0],
                                       rng=self.rng)
                                       
            # Append some random slices of the 
            i = 0
            while self.n < self.n_batch and i < self.num_samples_per_file:
                if self.complete_track_mode:
                    start = 0
                    end = start + self.desired_X_time_dim
                else:
                    start = np.random.randint(0, X_curr.shape[0] - \
                                                 self.desired_X_time_dim)
                    end = start + self.desired_X_time_dim
                self.X.append(X_curr[start:end, :])
                self.y.append(y_curr[start:end])
                self.bpms.append(self.bpm)
                self.fnames.append(self.curr_fname)
                i += 1
                self.n += 1
            print 'done with this wav'
            
    def downsample_wav(self, wav, ds):
        return wav.reshape(-1, ds).mean(axis=1)
    
    def append_raw_wav_chunks_and_target_vectors(self, wav):
            
        assert self.desired_X_time_dim % self.desired_X_raw_seconds == 0
        
        assert self.desired_rate % self.downsample == 0
        assert (self.desired_rate / self.downsample) % \
                (self.desired_X_time_dim / self.desired_X_raw_seconds) == 0
        
        frames_per_y = self.desired_X_time_dim
        
        # 
        y_fps = self.desired_X_time_dim / self.desired_X_raw_seconds
        X_fps = (self.desired_rate / self.downsample)

        # Length of the sample from the wav file we will use
        wav_complete_seconds = int(np.floor(len(wav) / self.desired_rate))
        
        # Cut wav to complete seconds
        wav = wav[0:(wav_complete_seconds * self.desired_rate)]
        # Downsample it
        X_curr = self.downsample_wav(wav, self.downsample)
        X_curr = X_curr.reshape(-1, 1)

        # Our divisor is how compressed our y representation is
        div = X_fps / y_fps

        # Target vector number of frames in total is the downsampled wav
        # length divided by our divisor        
        target_vec_num_frames = wav_complete_seconds * y_fps
        
        # Get a target vector of the same length representing the same
        # number of seconds
        y_curr = get_target_vector(self.bpm, wav_complete_seconds, 
                                   resolution=target_vec_num_frames,
                                   rng=self.rng)

        # This assertion should follow from the above I think but just to check
        assert X_curr.shape[0] % len(y_curr) == 0
        
        # Append some random slices of the 
        i = 0
        while self.n < self.n_batch and i < self.num_samples_per_file:
            if self.complete_track_mode:
                start = 0
                end = start + frames_per_y
            else:
                start = np.random.randint(0, len(y_curr) - frames_per_y)
                end = start + frames_per_y
            self.X.append(X_curr[(start*div):(end*div), :].astype(np.float32))
            self.y.append(y_curr[start:end].astype(np.float32))
            self.bpms.append(self.bpm)
            self.fnames.append(self.curr_fname)
            i += 1
            self.n += 1
        print 'done with this wav'
        
    
    def check_bpm(self, track_full_path):
        
        # Try to extract the BPM from the ID3 tag of an mp3
        try:
            self.bpm = int(get_track_bpm_from_id3_tag(track_full_path))
            
        # Return False if cannot do this for some reason
        except ValueError:
            print 'BPM read error in file', track_full_path
            return False
        except TypeError:
            print 'BPM read error in file', track_full_path
            return False
            
        print self.bpm, 'bpm'
        return True
    
    def get_next_track(self):
        # Pick a track
        #return np.random.choice(self.track_fnames, 1)[0]
    
        self.curr_track_idx += 1
        if self.curr_track_idx >= len(self.track_fnames):
            self.curr_track_idx = 0
            
        return self.track_fnames[self.curr_track_idx]
        
    def get_spectogram_training_set(self, n_batch=10):
        
        # Read in parameter and initialise count
        self.n_batch = n_batch
        self.n = 0
        
        self.num_samples_per_file = \
            max(1, int(np.ceil(self.n_batch * 1. / len(self.track_fnames))))

        # Initialise training data matrices
        self.X, self.y, self.bpms, self.fnames = [], [], [], []
        
        # Keep looping until we have the desired number of training samples
        while self.n < self.n_batch:
            
            self.curr_fname = self.get_next_track()          
            
            track_full_path = self.tracks_path + '/' + self.curr_fname
            
            # If the track has BPM information
            if self.check_bpm(track_full_path):
                # Get a normalised wav of the track
                wav = self.get_wav(track_full_path)
                
                if self.complete_track_mode:
                    #number of seconds in wav, rounded to multiple of 4 secs
                    self.desired_X_raw_seconds = \
                        int(self.desired_X_raw_seconds_orig * \
                            np.floor((len(wav)*1. / \
                        (self.desired_rate * \
                        self.desired_X_raw_seconds_orig))))
                    self.desired_X_time_dim = 441 * self.desired_X_raw_seconds
                
                # Append multiple clips of the wav the the training X matrix
                if wav is not None:
                    # If we provide this parameter, then we want raw wavs in X
                    # rather than FFTs
                    if self.desired_X_raw_seconds is not None:
                        self.append_raw_wav_chunks_and_target_vectors(wav)
                    else:
                        self.append_wav_to_Xy(wav)
                    
            print self.n, '/', self.n_batch, 'done'
        
        # Return X, y, bpms, fnames as np.arrays in desired shape for Keras
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        if not self.complete_track_mode:
            self.X = self.X.reshape(-1, 1, self.X.shape[1], self.X.shape[2])
        self.bpms = np.array(self.bpms)
        return (self.X, self.y, self.bpms, self.fnames)
        
def reverse_find(s, subs):
    return len(s) - s[::-1].find(subs)

def get_track_bpm_from_id3_tag(file_path):
    
    # Read BPM from ID3 tag
    print file_path
    if file_path.endswith('.mp3'):
        return Reader(file_path).getValue('TBPM')
        
    # Otherwise standard .wav file format should have BPM then space
    else:
        if '/' in file_path:
            return file_path[reverse_find(file_path, '/'):file_path.find(' ')]
        else:
            return file_path[0:file_path.find(' ')]

def get_fft(s, downsample=16):
    c = scifft(s)
    # you only need half of the fft list (real signal symmetry)
    d = len(c)/2
    return abs(c[:d])

def normalise(v):
    return (v - v.min()) / (v.max() - v.min())
def rmse(a, b):
    return np.sqrt(np.mean(np.square(a-b)))
def mae(a, b):
    return np.mean(np.abs(a-b))
    
def convert_an_mp3_to_wav(mp3_path, wav_path):
    import subprocess
    command = 'mpg123 -w ' + wav_path + ' "' + mp3_path + '"'
    subprocess.call(command, shell=True)

def get_target_vector(bpm, seconds, resolution, rng):
    
    # Initialise the output array with some small random noise
    target_vec = np.random.rand(resolution) * 0.001
    
    seconds_per_beat = 60. / bpm
    frames_per_second = resolution * 1. / seconds
    frames_per_beat = seconds_per_beat * frames_per_second
    num_complete_beats = int(np.floor(target_vec.shape[0] / frames_per_beat))
    
    # For each complete beat contained within the time this vector represents
    for i in xrange(num_complete_beats):
        # Set the entries where beats occur to 1
        pos = int(np.round(i * frames_per_beat))
        target_vec[pos] = 1.
        
        # We can add some padding around where the beats occur if we like...
        for j in xrange(-rng, rng):
            pos_new = pos + j
            if j != 0 and pos_new >= 0 and pos_new <= target_vec.shape[0]:
                target_vec[pos_new] = 1. / np.square(np.abs(j) + 1)
    
    return target_vec