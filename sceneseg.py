import numpy as np
np.random.seed(42)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
import torch
import glob
import sys
import os
import joblib
import pickle

# Where to get the data
PATH = os.path.join(os.getcwd(), "data_dir")

# Constants
FEATURES_DIM = 2048 + 512 + 512 + 512
MAX_MOVIE_LENGTH = 3100
NUM_EPOCHS = 10

def fetch_movies(path=PATH):
    """
    Load .pkl movie files
    
    Argument:
    ---------
    path -- string representing files path
    """
    filenames = glob.glob(os.path.join(PATH, "tt*.pkl"))
    movies = []
    for fn in filenames:
        try:
            with open(fn, 'rb') as fin:
                movies.append(pickle.load(fin))
        except EOFError:
            break
    return movies

def split_train_test(data, train_size=52):
    """
    Split data into train and test sets
    
    Argument:
    --------
    data -- a list of dictionaries each containing a movie information
    train_size -- integer representing the number of movies used for training
    """
    # For stable output across runs
    np.random.seed(42)
    # Shuffle indices
    shuffled_indices = np.random.permutation(len(data))
    train_indices = shuffled_indices[:train_size]
    test_indices = shuffled_indices[train_size:]
    train_set = [data[i] for i in train_indices]
    test_set = [data[i] for i in test_indices]
    return train_set, test_set

def transform_movies(movies, features=['place', 'cast', 'action', 'audio'], pad_len=MAX_MOVIE_LENGTH):
    """
    Unroll the given features by column and separate features from labels.
    Then pad the sequences in each movie to the length of the longest movie.
    
    Arguments:
    ----------
    movies -- a list of dictionaries each containing a movie information
    features -- list of string representing data features
    pad-len -- integer for the maximum length of a movie
    
    Return:
    -------
    X_padded -- a 2D numpy array
    Y_padded -- a 2D numpy array
    """
    X, Y = [], []
    # Unroll the features
    for movie in movies: 
        row = torch.cat([movie[feat] for feat in features], dim=1)
        X.append(row.numpy())
        # Pre-pad the label since its length is N-1
        labels = movie['scene_transition_boundary_ground_truth']
        labels = torch.cat([torch.tensor([False]), labels])
        Y.append(labels.numpy())
    # Pad the sequences
    X_padded = pad_sequences(X, maxlen=pad_len, padding='post', dtype='float32')
    Y_padded = pad_sequences(Y, value=False, maxlen=pad_len, padding='post')
    return X_padded, Y_padded

class myCallback(tf.keras.callbacks.Callback):
    """To stop training early once accuracy reach 95%"""
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.95):
            print("\nReached 95% accuracy, so cancelling training!")
            self.model.stop_training = True
            
##### Build LSTM Model #####
def build_lstm(n_layers=2, n_neurons=32, input_shape=[None, FEATURES_DIM]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for _ in range(n_layers):
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=n_neurons,
                                                               return_sequences=True)))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(1, activation='sigmoid')))
    model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])
    return model


##### Build WaveNet Model #####
def build_wave_net(input_shape=[None, FEATURES_DIM], num_blocks=2, num_layers=4, 
                   filters1=10, filters2=5, kern1=2, kern2=1, padding='same'):
    rates = [2**i for i in range(num_layers)]
    wave_model = keras.models.Sequential()
    wave_model.add(keras.layers.InputLayer(input_shape=input_shape))
    for rate in rates * num_blocks:
        wave_model.add(keras.layers.Conv1D(filters=filters1, 
                                              kernel_size=kern1, 
                                              padding='same',
                                              activation='relu', dilation_rate=rate))
    wave_model.add(keras.layers.Conv1D(filters=filters2, kernel_size=kern2))
    wave_model.add(keras.layers.Dense(1, activation='sigmoid'))
    wave_model.compile(loss='binary_crossentropy',
                       optimizer=keras.optimizers.RMSprop(lr=2e-5),
                       metrics=['accuracy'])
    return wave_model

##### Unpad predictions and write out files ######
def unpad_predictions(movies, yhat_probs):
    """
    Truncate the padded predictions to movie's original length
    
    Arguments:
    ----------
    movies -- a list of dictionaries containing movies information
    yhat_probs -- a 2D numpy array representing prediction for the given movies data set
    
    Return:
    -------
    yhat_dict -- a dictionary with each movie imbd_id as key and 
                 prediction probabilities as a 1D numpy array
    """
    imdb_lengths = [(movie['imdb_id'], movie['place'].shape[0]) for movie in movies]
    yhat_dict = dict()
    for (imdb, length), yhat in zip(imdb_lengths, yhat_probs):
        yhat = yhat[1:length]
        yhat_dict[imdb] = yhat
    return yhat_dict

def write_predictions(yhat_unpadded_dict, path=PATH):
    """
    Pickle the predictions
    
    Arguments:
    ----------
    yhat_unpadded_dict -- a dictionary of prediction consistent with the length of the ground-truth label
    path -- a string representing the files path
    """
    for imdb in yhat_unpadded_dict.keys():
        # Load existing pkl movie file
        filename = os.path.join(PATH, imdb + ".pkl")
        try:
            x = pickle.load(open(filename, "rb"))
            x['scene_transition_boundary_prediction'] = yhat_unpadded_dict[imdb].flatten()
            pickle.dump(x, open(filename, "wb"))
        except:
            break



if __name__ == '__main__':
    movies = fetch_movies()
    X, y = transform_movies(movies)
    # RandomSearchCV result: 3 blocks, 9 layers, filters1=13, filters2=10
    best_model = build_wave_net(num_blocks=3, num_layers=9, filters1=13, filters2=10)
    best_model.fit(X, y, epochs=20)
    yhat = best_model.predict(X)
    yhat_unpadded = unpad_predictions(movies, yhat)
    write_predictions(yhat_unpadded)


