# -*- coding: utf-8 -*-
'''
Neural Network for detecting beats in clips of audio
'''
from matplotlib import pyplot as plt
import numpy as np
import cPickle
from DataGen import *

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.optimizers import Adam
from multiprocessing.pool import ThreadPool

np.random.seed(1)  # for reproducibility

print("Loading data...")
X_train, y_train, bpms_train, fnames_train = \
    cPickle.load(open('Xy_train.dump', 'rb'))
X_val, y_val, bpms_val, fnames_val = \
    cPickle.load(open('Xy_vali.dump', 'rb'))

# Make data-generating objects for the training and validation data
gen_train = DataGen(X_train, y_train)
gen_val = DataGen(X_val, y_val)

# Specify and compile the neural network model
max_pool = 4

model = Sequential()
model.add(Convolution1D(4 * max_pool, 3, border_mode='same', 
                        input_shape=(gen_train.num_chunks, 
                                     gen_train.num_features)))
if max_pool > 1:
    model.add(Reshape((1764 * max_pool, 1)))
    model.add(MaxPooling1D(pool_length=max_pool))
model.add(Activation('relu'))
model.add(Flatten())
model.summary()
model.compile(loss='mse', optimizer=Adam())

'''
Rather than using Keras' fit_generator() function, I wrote my own function 
which essentially does the same thing, except the batch sizes can be bigger:
(this functionality appears to be absent in fit_generator())
'''
def fit_model(m, max_loops=50, patience=1):
    
    from time import sleep
    
    pool = ThreadPool(processes=2)
    
    num_epochs_per_loop = 1
    
    val_size = 96 * 2
    train_size = 96 * 8
    
    best_loss = 99999.
    count_for_patience = 0
    histories = []
    
    # Initial run training and validation sets (future runs will be generated
    # in the background while training is underway)
    Xv, yv = gen_val.gen_dataset(val_size)
    Xt, yt = gen_train.gen_dataset(train_size)
    
    for i in range(max_loops):
        
        thread_val = pool.apply_async(gen_val.gen_dataset, ([val_size]))
        thread_train = pool.apply_async(gen_train.gen_dataset, ([train_size]))
        
        histories.append(m.fit(Xt, yt,
                         batch_size=96, 
                         nb_epoch=num_epochs_per_loop,
                         verbose=1, 
                         validation_data=(Xv, yv),
                         shuffle=False))
    
        while not (thread_val.ready() and thread_train.ready()):
            print 'still waiting to produce training data (bad!)'
            sleep(1)
            
        Xv, yv = thread_val.get()
        Xt, yt = thread_train.get()
        
        # Get loss and see if it has improved. Early stop if necessary.
        loss = history.history['val_loss'][0]
        if loss < best_loss:
            best_loss = loss
            count_for_patience = 0
        else:
            count_for_patience += 1
        if count_for_patience > (patience + 1):
            print 'Validation loss did not improve after', \
                   patience, 'loops: early stopping.'
            break
        
    return histories, m

history, model = fit_model(model)

# Plot the performance over new random samples 
# from the training and validation wavs
def plot_predicted_vs_actual_pulses(gen, num, model, X_o=None):
    X, y = gen.gen_dataset(num)
    Xplot = X.reshape(X.shape[0], -1, gen.div).mean(axis=2)
    for i in range(0, y.shape[0], max(y.shape[0] / num, 1)):
        plt.figure(figsize=(12,10))
        if X_o is None:
            pred = model.predict(X[i:(i+1)])
        else:
            pred = model.predict(X_o[i:(i+1)])
        plt.plot(Xplot[i] / 2 + 0.5, alpha=0.5)
        plt.plot(y[i, :] * 9999 - 9990., 'black', linewidth=4, alpha=0.7)
        plt.plot(pred[0], 'red', linewidth=3)
        plt.ylim([-0.2,1.2])
        plt.show()

num_results_to_show = 10
print '\n\n\ntraining performance...'
plot_predicted_vs_actual_pulses(gen_train, num_results_to_show, model)
print '\n\n\nvalidation performance...'
plot_predicted_vs_actual_pulses(gen_val, num_results_to_show, model)