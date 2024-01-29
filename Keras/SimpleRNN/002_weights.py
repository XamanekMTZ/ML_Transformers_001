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
print('\n')

x = np.array([1, 2, 3])

# Reshape the input to the required sample_size x time_steps x features
x_input = np.reshape( x, (1, 3, 1 ) )

# Predict the output
y_pred_model = demo_model.predict(x_input)


# Compute the output manually
m = 2
# Hidden layer initialized to zeros
h0 = np.zeros( m )

# first hidden layer
h1 = np.dot( x[0], wx ) + h0 + bh

# second hidden layer
h2 = np.dot( x[1], wx ) + np.dot( h1, wh ) + bh

# third hidden layer
h3 = np.dot( x[2], wx ) + np.dot( h2, wh ) + bh

# output layer
o3 = np.dot( h3, wy ) + by

print('\n Hidden Layer 1\n h1 = ', h1)
print('\n Hidden Layer 2\n h2 = ', h2)
print('\n Hidden Layer 3\n h3 = ', h3)
print('Prediction from network ', y_pred_model)
print('Prediction from our computation ', o3)
print('\n')