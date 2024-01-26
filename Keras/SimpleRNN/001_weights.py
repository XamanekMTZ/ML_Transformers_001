import math

from pandas import read_csv
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def create_RNN( hidden_units, dense_units, input_shape, activation ):
  model = Sequential()
  model.add(SimpleRNN( 
    hidden_units, input_shape = input_shape, activation = activation[0] 
    ) )
  model.add( Dense( units = dense_units, activation = activation[1] ) )
  model.compile( loss = 'mean_squared_error', optimizer = 'adam' )
  return model

demo_model = create_RNN( 2, 1, ( 3, 1 ), activation = [ 'linear', 'linear' ] )

wx = demo_model.get_weights()[0]
wh = demo_model.get_weights()[1]
bh = demo_model.get_weights()[2]
wy = demo_model.get_weights()[3]
by = demo_model.get_weights()[4]

print('\n wx = ', wx, '\n wh = ', wh, '\n bh = ', bh, '\n wy = ', wy, '\n by = ', by)
