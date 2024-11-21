# Keras regression example

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.datasets import boston_housing

# load dataset
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

model = Sequential()
model.add(Dense(16, input_dim=13, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1)) 

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, epochs=200, batch_size=10)

Y_prediction = model.predict(X_test).flatten()

for i in range(10):
  real_price = y_test[i]
  predicted_price = Y_prediction[i]
  print('Real Price: {:.3f}, Predicted Price: {:.3f}'.format(real_price,
                                                             predicted_price))