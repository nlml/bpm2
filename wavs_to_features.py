# -*- coding: utf-8 -*-
"""
Creates training and validation set features from wavs placed in subdirectory 
TRACKS_PATH, and pickles these features for training.

*******************************************************************************

WAVS IN THE SUBDIRECTORY MUST BE OF FORMAT '120 NAME...' where 120 is the 
BPM - in other words: '*Integer BPM in digits*[space]*everything else*'

WAVS MUST ALSO START EXACTLY ON THE FIRST BEAT OF A BAR!

*******************************************************************************

Some help from:
http://stackoverflow.com/a/23378284/6167850

"""
from os import listdir
import numpy as np
from matplotlib import pyplot as plt
import cPickle
from ExtractFeatures import *

# Audio files are stored in this relative path
TRACKS_PATH = 'wavs'

# Randomly shuffle the wav files
np.random.seed(1)
all_tracks = listdir(TRACKS_PATH)
np.random.shuffle(all_tracks)

# 20% of the wav's we have will be never seen during training
propn_vali_tracks = 0.2
num_vali_tracks = int(len(all_tracks) * propn_vali_tracks)

# Get training set features from wavs
tracks_subset = all_tracks[0:-num_vali_tracks]
ns = WavFeatureExtractor(downsample=4,
                         desired_X_time_dim=441*4,
                         track_fnames=tracks_subset,
                         tracks_path=TRACKS_PATH,
                         desired_X_raw_seconds=4,
                         complete_track_mode=True)
X, y, bpms, fnames = ns.get_spectogram_training_set(n_batch=len(tracks_subset))
# Save features to disk
cPickle.dump((X, y, bpms, fnames), open('Xy_train.dump', 'wb'))

# Get test set features from remaining wavs and save to disk
tracks_subset = all_tracks[-num_vali_tracks:]
ns.track_fnames = tracks_subset
X, y, bpms, fnames = ns.get_spectogram_training_set(len(tracks_subset))
cPickle.dump((X, y, bpms, fnames), open('Xy_vali.dump', 'wb'))