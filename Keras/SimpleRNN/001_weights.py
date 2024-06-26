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

# Weights input units
wx = demo_model.get_weights()[0]
# Weights hidden units
wh = demo_model.get_weights()[1]
# Bias hidden units
bh = demo_model.get_weights()[2]
# Weights output units
wy = demo_model.get_weights()[3]
# Bias output units
by = demo_model.get_weights()[4]

print('\n Weights input units\n wx = ', wx)
print('\n Weights hidden units\n wh = ', wh)
print('\n Bias hidden units\n bh = ', bh)
print('\n Weights output units\n wy = ', wy)
print('\n Bias output units\n by = ', by)
