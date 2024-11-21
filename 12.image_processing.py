# simple image processing

from skimage import io 
from tkinter.filedialog import askopenfilename  # choose file
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg 
# %matplotlib inline

fname = askopenfilename()           # choose image file
image = mpimg.imread(fname)         # read image
plt.imshow(image)                   # display image 
plt.show()

# type of image object
type(image)

# image shape
print(image.shape)

# image data
print(image[:,:,1])  # red channel


# color to gray
from skimage import color 

gray_image = color.rgb2gray(image)
print(gray_image.shape)
plt.imshow(gray_image, cmap='gray')  
plt.show()

# resize
from skimage import transform 

new_shape = (image.shape[0] // 2, image.shape[1] // 2, 
             image.shape[2])
small = transform.resize(image=image, 
        output_shape=new_shape)

print(small.shape)
plt.imshow(small)
plt.show()
              

# filp, rotate
from skimage.transform import rotate

plt.imshow(np.fliplr(image))  # filp
plt.imshow(rotate(image, angle=45))  # rotate
plt.show()


# filters
from skimage.filters import sobel_h
plt.imshow(sobel_h(gray_image))
plt.show()

# save image
io.imsave("d:/data/test2.png", image)



                  