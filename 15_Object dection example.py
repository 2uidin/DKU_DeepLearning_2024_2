# Object dection example
# Requirements:
# pip install keras-cv
# pip install opencv-python

import os

os.environ["KERAS_BACKEND"] = "tensorflow"  # @param ["tensorflow", "jax", "torch"]

import keras
import keras_cv
import numpy as np

from keras_cv import visualization
import matplotlib.pyplot as plt

# Load a pretrained model
pretrained_model = keras_cv.models.YOLOV8Detector.from_preset(
    "yolo_v8_m_pascalvoc", bounding_box_format="xywh"
)

# Load an image from a URL or local path
filepath = "C:/Users/SIM/DKU_DL/cat.png"
image = keras.utils.load_img(filepath)
image = np.array(image)

visualization.plot_image_gallery(
    np.array([image]),
    value_range=(0, 255),
    rows=1,
    cols=1,
    scale=5,
)
plt.show()

# resize image
inference_resizing = keras_cv.layers.Resizing(
    640, 640, pad_to_aspect_ratio=True, bounding_box_format="xywh"
)

image_batch = inference_resizing([image])

class_ids = [
    "Aeroplane",  "Bicycle", "Bird",  "Boat", "Bottle",
    "Bus",  "Car", "Cat", "Chair", "Cow", "Dining Table",
    "Dog", "Horse", "Motorbike", "Person", "Potted Plant",
    "Sheep", "Sofa", "Train", "Tvmonitor", "Total", ]
class_mapping = dict(zip(range(len(class_ids)), class_ids))

y_pred = pretrained_model.predict(image_batch)
# y_pred is a bounding box Tensor:
# {"classes": ..., boxes": ...}
print(y_pred['classes'][0][:4]) # 어떤 클래스인가 → class_ids 에서 추출
print(y_pred['confidence'][0][:4])  # 각 개체 당 정확도는 얼마인가
print(y_pred['boxes'][0][:4])   # 각 개체의 박스 위치 → xywh 순서

visualization.plot_bounding_box_gallery(
    image_batch,
    value_range=(0, 255),
    rows=1,
    cols=1,
    y_pred=y_pred,
    scale=5,
    font_scale=0.7,
    bounding_box_format="xywh",
    class_mapping=class_mapping,
)
# plt.savefig('default.png')    # '파일명.확장자'로 결과 이미지를 저장
plt.show()

