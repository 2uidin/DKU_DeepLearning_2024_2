# Augmentation example

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img   # type: ignore

datagen = ImageDataGenerator(
rotation_range=40, # 0~180
width_shift_range=0.2,
height_shift_range=0.2,
rescale=1./255, # 픽셀값을 0~1 로 변환
shear_range=0.2, # shearing transformations
zoom_range=0.2, # randomly zooming
horizontal_flip=True, # randomly flipping
fill_mode='nearest') # filling in newly created pixels

img = load_img('C:/Users/SIM/DKU_DL/cat.png')
x = img_to_array(img) # shape (3, 331, 237)
x = x.reshape((1,) + x.shape) # shape (1, 3, 331, 237)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `d:/data/aug/` directory
i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir='C:/Users/SIM/DKU_DL/aug/', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 30:
        break # or for working infinitely 