# -*- coding: utf-8 -*-
"""
Class to generate mini-batches of training data from full-length .wav features.
It will randomly sample 4 second subsets from the full-length audio.

@author: liam
"""
import numpy as np

# Draw a triangle around the beat pulses
def pad_y(y, rng):
    if rng > 0:
        r = 1. / (rng + 1)
        indices = np.where(y==1.)[0]
        for j in range(-rng, rng):
            curr_indices = np.minimum(np.maximum(indices + j, 0), len(y) - 1)
            y[curr_indices] = max(0., 1. - r * np.abs(j))
        y[np.where(y<0.05)[0]] = 0.
    return y

class DataGen():

    def __init__(self, X, y, X_sample_len=44100, num_chunks=441, y_pad=10):
        assert X_sample_len % num_chunks == 0
        self.X_sample_len = X_sample_len
        self.num_chunks = num_chunks
        self.num_features = self.X_sample_len / self.num_chunks
        self.y_pad = y_pad
        self.set_Xy(X, y)
        
    def set_Xy(self, X, y):
        assert X[0].shape[0] % y[0].shape[0] == 0
        self.X = X
        self.y = y
        self.div = self.X[0].shape[0] / self.y[0].shape[0]
        self.y_sample_len = self.X_sample_len / self.div
        self.min_length = \
            min([i.shape[0] / (self.X_sample_len) for i in self.X])
        
    def get_single_sample(self, choice):
        start = np.random.randint(0, self.X[choice].shape[0] - \
                                     self.X_sample_len + 1)
        end = start + self.X_sample_len
        return self.get_sample_from_start_to_end(choice, start, end)
        
    def get_sample_from_start_to_end(self, choice, start, end):
        return self.X[choice][start:end, :].reshape(self.num_chunks, -1), \
                self.y[choice][start / self.div:end / self.div]
        
    def gen_dataset(self, size):
        X_out, y_out = [], []
        for i in range(size):
            choice = np.random.randint(0, len(self.X))
            samp = self.get_single_sample(choice)
            X_out.append(samp[0])
            if self.y_pad:
                y_out.append(pad_y(samp[1], self.y_pad))
            else:
                y_out.append(samp[1])
        return np.array(X_out), np.array(y_out)
    
    def gen_seq_sample_sliced(self, t, size, offset):
        X_out, y_out = [], []
        start = offset * self.X_sample_len
        end = start + size * self.X_sample_len
        for i in range(start, end, self.X_sample_len):
            samp = self.get_sample_from_start_to_end(t, i, i + self.X_sample_len)
            X_out.append(samp[0])
            y_out.append(samp[1])
        return np.array(X_out), np.array(y_out)
    
    def gen_sequential_sample(self, t, size, offset=0):

        if offset is None:
            smax = int((self.X[t].shape[0] - size * self.X_sample_len) / \
                    self.X_sample_len) + 1
            offset = np.random.randint(0, smax)
        samp = self.gen_seq_sample_sliced(t, size, offset)
        return samp[0], samp[1]
            
    def gen_sequential_training_set(self, size=None, track_idxs=None, size_per=1):
        
        X_out, y_out = [], []
        
        if track_idxs is None:
            track_idxs = range(len(self.X))
            
        if size is None:
            size = self.min_length
        
        for idx, t in enumerate(track_idxs):
            for i in range(size_per):
                if size_per > 1:
                    samp = self.gen_sequential_sample(t, size, offset=None)
                else:
                    samp = self.gen_sequential_sample(t, size)
                X_out.append(samp[0])
                y_out.append(samp[1])
        return np.array(X_out), np.array(y_out)
    
    def flow(self):
        while 1:
            X, y = self.get_single_sample(np.random.randint(0, len(self.X)))
            #X, y = self.gen_dataset(96)
            yield X.reshape(1, self.num_chunks, -1), y.reshape(1, -1)