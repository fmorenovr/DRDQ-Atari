from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, LSTM
from keras.layers.wrappers import TimeDistributed

learning_rate = 0.00025

class ConvolutionalNeuralNetwork:
  def __init__(self, input_shape, action_space, mode):
    print("The input shape is: %d", input_shape)
    self.model = Sequential()
    if mode == "mse":
      self.model.add(Conv2D(32, (8, 8), strides=(4, 4), padding="valid", activation="relu", input_shape=input_shape, data_format="channels_first"))
      self.model.add(Conv2D(64, (4, 4), strides=(2, 2), padding="valid", activation="relu", input_shape=input_shape, data_format="channels_first"))
      self.model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="valid", activation="relu", input_shape=input_shape, data_format="channels_first"))
      self.model.add(Flatten())
      self.model.add(Dense(512, activation="relu"))
      self.model.add(Dense(action_space, activation='linear'))
      self.model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=learning_rate, rho=0.95, epsilon=0.01), metrics=["accuracy"])
    elif mode == "adam":
      self.model.add(Conv2D(32, (8, 8), strides=(4, 4), padding="valid", activation="relu", input_shape=input_shape, data_format="channels_first"))
      self.model.add(Conv2D(64, (4, 4), strides=(2, 2), padding="valid", activation="relu", input_shape=input_shape, data_format="channels_first"))
      self.model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="valid", activation="relu", input_shape=input_shape, data_format="channels_first"))
      self.model.add(Flatten())
      self.model.add(Dense(512, activation="relu"))
      self.model.add(Dense(action_space, activation='linear'))
      self.model.compile(loss='mse',optimizer=Adam(lr=learning_rate))
    self.model.summary()

class RecurrentConvolutionalNeuralNetwork:
  def __init__(self, input_shp, action_space, mode):
    input_shape = input_shp + (1,)
    print("The input shape is: %d", input_shape)
    self.model = Sequential()
    if mode == "mse":
      #self.model.add(TimeDistributed(Conv2D(32, 8, strides=(4, 4), padding="valid", activation='relu', input_shape=input_shape, data_format="channels_first")))
      self.model.add(TimeDistributed(Conv2D(32, (8, 8), strides=(4, 4), padding="valid", activation='relu'),  input_shape=input_shape))
      #self.model.add(TimeDistributed(Conv2D(64, 4, strides=(2, 2), padding="valid", activation="relu", input_shape=input_shape, data_format="channels_first")))
      self.model.add(TimeDistributed(Conv2D(64, (4, 4), strides=(2, 2), padding="valid", activation='relu'), input_shape=input_shape))
      #self.model.add(TimeDistributed(Conv2D(64, 3, strides=(1, 1), padding="valid", activation="relu", input_shape=input_shape, data_format="channels_first")))
      self.model.add(TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), padding="valid", activation='relu'), input_shape=input_shape))
      self.model.add(TimeDistributed(Flatten()))
      self.model.add(LSTM(512))
      self.model.add(Dense(128, activation='relu'))
      self.model.add(Dense(action_space, activation='linear'))
      self.model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=learning_rate, rho=0.95, epsilon=0.01), metrics=["accuracy"])
    elif mode == "adam":
      #self.model.add(TimeDistributed(Conv2D(32, 8, strides=(4, 4), padding="valid", activation='relu', input_shape=input_shape, data_format="channels_first")))
      self.model.add(TimeDistributed(Conv2D(32, (8, 8), strides=(4, 4), padding="valid", activation='relu'), input_shape=input_shape))
      #self.model.add(TimeDistributed(Conv2D(64, 4, strides=(2, 2), padding="valid", activation="relu", input_shape=input_shape, data_format="channels_first")))
      self.model.add(TimeDistributed(Conv2D(64, (4, 4), strides=(2, 2), padding="valid", activation='relu'), input_shape=input_shape))
      #self.model.add(TimeDistributed(Conv2D(64, 3, strides=(1, 1), padding="valid", activation="relu", input_shape=input_shape, data_format="channels_first")))
      self.model.add(TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), padding="valid", activation='relu'), input_shape=input_shape))
      self.model.add(TimeDistributed(Flatten()))
      self.model.add(LSTM(512,  activation='tanh'))
      self.model.add(Dense(action_space, activation='linear'))
      self.model.compile(loss='mse',optimizer=Adam(lr=learning_rate))
    self.model.summary()
