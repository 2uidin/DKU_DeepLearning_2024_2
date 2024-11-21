# CNN MNIST example

from keras.datasets import mnist        # type: ignore
from keras.models import Sequential     # type: ignore
from keras.layers import Dense, Input   # type: ignore
from keras.layers import Dropout        # type: ignore
from keras.layers import Flatten        # type: ignore
from keras.layers import Conv2D         # type: ignore
from keras.layers import MaxPooling2D   # type: ignore
from keras.utils  import to_categorical # type: ignore
import numpy as np
import matplotlib.pyplot as plt

# define image size
img_rows=28
img_cols=28

# load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

# reshape
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)


# one hot encoded
y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)

# fix random seed for reproducibility 
seed = 100 
np.random.seed(seed)
num_classes = 10

# create CNN model
def cnn_model():
    # define model
    model = Sequential()
    model.add(Input(shape=(img_rows, img_cols, 1)))
    model.add(Conv2D(32, kernel_size=(5, 5), 
                            padding='valid',
                            strides=(1, 1),
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(127, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    return model


# build the model
model = cnn_model()

# Fit the model
disp = model.fit(X_train, y_train, 
          validation_data=(X_test, y_test), 
          epochs=10, 
          batch_size=200, 
          verbose=1)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("loss: %.2f" % scores[0])
print("acc: %.2f" % scores[1])

# summarize history for accuracy
plt.plot(disp.history['accuracy'])
plt.plot(disp.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
