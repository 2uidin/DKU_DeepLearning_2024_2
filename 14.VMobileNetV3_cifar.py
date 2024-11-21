# CIFIR classification using MobileNetV3

#######################################################
## transfer learning

from keras import optimizers
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras.utils import np_utils
from keras.applications import MobileNetV3Small
import keras.backend as K

K.clear_session()

# set up base model
img_width, img_height = 32, 32
base_model = MobileNetV3Small(weights='imagenet', 
                   include_top=False,        # output 부분 사용x
                   input_shape= (img_width, img_height, 3)) 

base_model.trainable = False

nb_epoch = 10      # try 50
nb_classes = 10

# load dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

# define model
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(1024,activation=('relu')))
model.add(Dense(512,activation=('relu'))) 
model.add(Dense(256,activation=('relu'))) 
model.add(Dense(128,activation=('relu')))
model.add(Dense(10,activation=('softmax')))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(learning_rate=1e-2),
              metrics=['accuracy'])


model.fit(X_train, y_train, 
          validation_data=(X_test, y_test), 
          epochs=nb_epoch, 
          batch_size=200, 
          verbose=1)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("loss: %.2f" % scores[0])
print("acc: %.2f" % scores[1])