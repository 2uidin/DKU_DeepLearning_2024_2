# keras DNN example
# dataset : MNIST

# load required modules
from keras.datasets import mnist        # type: ignore
from keras import optimizers
from keras.models import Sequential     # type: ignore
from keras.layers import Dense, Input   # type: ignore
from keras.layers import Flatten        # type: ignore
from keras.layers import Dropout        # type: ignore
from keras.utils import to_categorical  # type: ignore

import matplotlib.pyplot as plt
import numpy as np

import sys
import os
sys.path.append(os.path.abspath("C:/Users/SIM/DKU_DL/TrainPlot.py"))
import TrainPlot              # call TrainPlot.py  

# load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# one hot encoded
y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)

################################################################
# define model (DNN structure)
epochs = 20
batch_size = 128
learning_rate = 0.01

model = Sequential()
model.add(Input(shape=(28,28)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate = 0.4))
model.add(Dense(128, activation='relu'))
model.add(Dropout(rate = 0.3))
model.add(Dense(64, activation='relu')) # 64개의 노드를 가지는 hidden layer 추가
model.add(Dropout(rate = 0.2))
model.add(Dense(10, activation='softmax'))

model.summary()  # show model structure

# Compile model
adam = optimizers.Adam(learning_rate=learning_rate)
model.compile(loss='categorical_crossentropy', 
              optimizer=adam, 
              metrics=['accuracy'])

# model fitting (learning)
disp = model.fit(x_train, y_train, 
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1,        # print fitting process  
                 validation_split = 0.2,
                #  callbacks=[TrainPlot.TrainingPlot()]
                #   → 얘 실시간으로 그래프 그리는 거 안됨..
                 )

############################################################## 
# Test model
pred = model.predict(x_test)
print(pred)
y_classes = [np.argmax(y, axis=None, out=None) for y in pred]
print(y_classes)   # result of prediction

# model performance
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# summarize history for accuracy
plt.plot(disp.history['accuracy'])
plt.plot(disp.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(disp.history['loss'])
plt.plot(disp.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()



############################################################## 
# # model weights
# for lay in model.layers:
#     print(lay.name)
#     print(lay.get_weights())
    



