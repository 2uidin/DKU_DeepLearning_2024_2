# Predict a random image using VGG16

from keras.preprocessing.image import load_img          # type: ignore
from keras.preprocessing.image import img_to_array      # type: ignore
from keras.applications.vgg16 import preprocess_input   # type: ignore
from keras.applications.vgg16 import decode_predictions # type: ignore
from keras.applications.vgg16 import VGG16              # type: ignore

# load the model
model = VGG16()       # take a long time 

# load an image from file
image = load_img('C:/Users/SIM/DKU_DL/cat.png', target_size=(224, 224))

# convert the image pixels to a numpy array
image = img_to_array(image)

# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

# prepare the image for the VGG model
image = preprocess_input(image)

# predict the probability across all output classes
pred = model.predict(image)

# convert the probabilities to class labels
label = decode_predictions(pred)
print(label)

# retrieve the most likely result, e.g. highest probability
label = label[0][0]

# print the classification
print('%s (%.2f%%)' % (label[1], label[2]*100))
