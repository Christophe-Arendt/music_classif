#----------------------------------------------------------------------------
#   CNN - Model
#----------------------------------------------------------------------------
from utils import get_feature_stats, plot_history
import numpy as np
import os
import json

# DL
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers

# Data viz
import matplotlib.pyplot as plt

# ML
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join('..','dataset','data_10.json')
FIG_SIZE = (15,10)

#----------------------------------------------------------------------------


def load_data(data_path):
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    print('# Data loaded')
    return X, y



def prepare_datasets(test_size, validation_size):
    """Loads data and splits it into train, validation and test sets.
    :param test_size (float): Value in [0, 1] indicating percentage of data set to allocate to test split
    :param validation_size (float): Value in [0, 1] indicating percentage of train set to allocate to validation split
    :return X_train (ndarray): Input training set
    :return X_validation (ndarray): Input validation set
    :return X_test (ndarray): Input test set
    :return y_train (ndarray): Target training set
    :return y_validation (ndarray): Target validation set
    :return y_test (ndarray): Target test set
    """

    # load data
    X, y = load_data(DATA_PATH)

    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # add an axis to input sets
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    print('# Dataset preapred')
    return X_train, X_validation, X_test, y_train, y_validation, y_test

def model(input_shape, model_save = False):
  """Generates CNN model
  :param input_shape (tuple): Shape of input set
  :return model: CNN model
  """

  # build network topology
  model = models.Sequential()

  # 1st conv layer
  model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
  model.add(BatchNormalization())
  model.add(layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
  #model.add(layers.BatchNormalization())

  # 2nd conv layer
  model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.04)))
  model.add(layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
  #model.add(layers.BatchNormalization())

  # 3rd conv layer
  model.add(layers.Conv2D(32, (2, 2), activation='relu'))
  #model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
  #model.add(layers.BatchNormalization())

  # flatten output and feed it into dense layer
  model.add(layers.Flatten())
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dropout(0.3))

  # output layer
  model.add(layers.Dense(10, activation='softmax'))


  #------------------------------------------

  # compiler
  optimiser = Adam(learning_rate=0.0005)
  model.compile(optimizer=optimiser,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
  es = EarlyStopping(patience=5)
  print('# Model build')
  #------------------------------------------

  history = model.fit(X_train, y_train,
                      validation_data=(X_validation, y_validation),
                      batch_size=32,
                      epochs=50,
                     callbacks=[es],
                     verbose =1)

  if model_save == True:
    model.save(os.path.join('..','models','dl','cnn'))
    print('# Model saved')

  return model, history


def predict(model, X, y):
    """Predict a single sample using the trained model
    :param model: Trained classifier
    :param X: Input data
    :param y (int): Target
    """

    # add a dimension to input data for sample - model.predict() expects a 4d array in this case
    X = X[np.newaxis, ...] # array shape (1, 130, 13, 1)

    # perform prediction
    prediction = model.predict(X)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)

    print("Target: {}, Predicted label: {}".format(y, predicted_index))


#----------------------------------------------------------------------------
#   Execute
#----------------------------------------------------------------------------

if __name__ == '__main__':
  print('Run model')
  X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)
  input_shape = (X_train.shape[1], X_train.shape[2], 1)
  model, history = model(input_shape, model_save = True)
  # evaluate model
  print('Evaluate model')
  test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
  print('\nTest accuracy:', test_acc)
  plot_history(history)
  plt.show()




