# Import libraries and modules for data analysis
import pandas as pd

# Import models and utility functions from sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the Dataset
real_state_data = pd.read_csv( 
  '/home/xamanek/PythonProjects/ML_Transformers_001/Datasets/20240131a_Real_Estate.csv' 
  )

# Select features and target variable
features = [
  'Distance to the nearest MRT station', 
  'Number of convenience stores', 
  'Latitude', 
  'Longitude'
  ]
target = 'House price of unit area'

# Create the feature matrix X and target vector y
X = real_state_data[ features ]
y = real_state_data[ target ]

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split( 
  X, y, test_size = 0.2, random_state = 42 
  )

# Model initialization 
model = LinearRegression()

# Training the model
model.fit( X_train, y_train )