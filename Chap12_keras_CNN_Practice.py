from keras.models import Sequential                                 # type: ignore
from keras.layers import Dense, Input                               # type: ignore
from keras.layers import Flatten                                    # type: ignore
from keras.layers import Conv2D                                     # type: ignore
from keras.layers import MaxPooling2D                               # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# 디렉토리가 mnist처럼 예쁘게 되어있는 것이 아니기 때문에 경로를 손봐줘야 함.

# mnist
# └ train   (28*28, 60000 img)
# └ test    (28*28, 10000 img)
# 참조: https://lheon.tistory.com/60

# Shoe vs Sandal vs Boot Dataset
# └ Boot    (136*102, 5000 img)
# └ Sandal  (136*102, 5000 img)
# └ Shoe    (136*102, 5000 img)

path = 'C:/Users/SIM/Downloads/archive/Shoe vs Sandal vs Boot Dataset/'

# for img_class in os.listdir(path):
#     print(img_class)

# img_data = tf.keras.utils.image_dataset_from_directory(path)

img_gen = ImageDataGenerator(rescale=1./255,
                             validation_split = 0.3
                             )
# 참조: https://all4null.tistory.com/12
# 참조: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator

train = img_gen.flow_from_directory(path,                       # 경로 설정
                                    batch_size = 32,            # 원본에서 가져올 데이터 사이즈
                                    target_size = (256, 256),   # 사용할 CNN 모델에 맞게 resize
                                    class_mode = "categorical", # One-hot 인코딩 적용
                                    subset = "training"
                                    )

test = img_gen.flow_from_directory(path,
                                   batch_size = 32,
                                   target_size = (256, 256),
                                   class_mode = "categorical",
                                   subset = "validation"
                                   )
# 참조: https://techblog-history-younghunjo1.tistory.com/261

# fix random seed for reproducibility 
seed = 1234
np.random.seed(seed)
num_classes = 3

# create CNN model
def cnn_model():
    # define model
    model = Sequential()
    model.add(Input(shape=(256, 256, 3)))
    model.add(Conv2D(12, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(20, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    return model


# build the model
model = cnn_model()

# Fit the model
disp = model.fit(train, 
          validation_data=(test), 
          epochs=10, 
          batch_size=200, 
          verbose=1)

# Final evaluation of the model
scores = model.evaluate(test, verbose=0)
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

# Epoch 1/10
# 329/329 ━━━━━━━━━━━━━━━━━━━━ 160s 480ms/step - accuracy: 0.8051 - loss: 0.4590 - val_accuracy: 0.9344 - val_loss: 0.1779
# Epoch 2/10
# 329/329 ━━━━━━━━━━━━━━━━━━━━ 122s 370ms/step - accuracy: 0.9527 - loss: 0.1384 - val_accuracy: 0.9602 - val_loss: 0.1133
# Epoch 3/10
# 329/329 ━━━━━━━━━━━━━━━━━━━━ 94s 285ms/step - accuracy: 0.9816 - loss: 0.0548 - val_accuracy: 0.9469 - val_loss: 0.1628
# Epoch 4/10
# 329/329 ━━━━━━━━━━━━━━━━━━━━ 95s 288ms/step - accuracy: 0.9848 - loss: 0.0443 - val_accuracy: 0.9538 - val_loss: 0.1875
# Epoch 5/10
# 329/329 ━━━━━━━━━━━━━━━━━━━━ 95s 288ms/step - accuracy: 0.9894 - loss: 0.0292 - val_accuracy: 0.9444 - val_loss: 0.2148
# Epoch 6/10
# 329/329 ━━━━━━━━━━━━━━━━━━━━ 108s 329ms/step - accuracy: 0.9939 - loss: 0.0187 - val_accuracy: 0.9591 - val_loss: 0.1685
# Epoch 7/10
# 329/329 ━━━━━━━━━━━━━━━━━━━━ 97s 294ms/step - accuracy: 0.9973 - loss: 0.0091 - val_accuracy: 0.9527 - val_loss: 0.2111
# Epoch 8/10
# 329/329 ━━━━━━━━━━━━━━━━━━━━ 99s 301ms/step - accuracy: 0.9931 - loss: 0.0196 - val_accuracy: 0.9593 - val_loss: 0.1951
# Epoch 9/10
# 329/329 ━━━━━━━━━━━━━━━━━━━━ 99s 300ms/step - accuracy: 0.9937 - loss: 0.0218 - val_accuracy: 0.9573 - val_loss: 0.2328
# Epoch 10/10
# 329/329 ━━━━━━━━━━━━━━━━━━━━ 98s 299ms/step - accuracy: 0.9959 - loss: 0.0189 - val_accuracy: 0.9629 - val_loss: 0.2012
# loss: 0.20
# acc: 0.96