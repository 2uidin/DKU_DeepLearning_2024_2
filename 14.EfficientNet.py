# # CIFIR classification using EfficientNet
# https://pypi.org/project/efficientnet/#installation

from keras import optimizers
from keras.datasets import cifar10
from keras.engine import Model
from keras.layers import Dropout, Flatten, Dense
from keras.utils import np_utils

import efficientnet.keras as efn 

#model = efn.EfficientNetB0(weights='imagenet') 
img_width, img_height = 32, 32
base_model = efn.EfficientNetB0(weights='imagenet', 
                   include_top=False, 
                   input_shape=(32, 32, 3))

nb_epoch = 2    # 50 is good
nb_classes = 10

# load dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

# Extract the last layer from third block of vgg16 model
last = base_model.get_layer('top_activation').output

# Add classification layers on top of it
x = Flatten()(last)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(10, activation='softmax')(x)

model = Model(base_model.input, output)

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])

model.summary()


model.fit(X_train, y_train, 
          validation_data=(X_test, y_test), 
          nb_epoch=nb_epoch, 
          batch_size=200, 
          verbose=1)


# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("loss: %.2f" % scores[0])
print("acc: %.2f" % scores[1])
