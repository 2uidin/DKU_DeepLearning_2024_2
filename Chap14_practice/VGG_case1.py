from keras.preprocessing.image import load_img          # type: ignore
from keras.preprocessing.image import img_to_array      # type: ignore
from keras.applications.vgg16 import preprocess_input   # type: ignore
from keras.applications.vgg16 import decode_predictions # type: ignore
from keras.applications.vgg16 import VGG16              # type: ignore
import numpy as np

# VGG16 모델 로드
model = VGG16()

def classify_image(file_path):
    # 이미지 로드 및 전처리
    image = load_img(file_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    # 예측
    pred = model.predict(image)
    label = decode_predictions(pred)

    # 결과 반환
    result = []
    result.append('Classification result')
    result.append('--------------------')
    for i in range(3):
        result.append(f"{label[0][i][1]} \t ({label[0][i][2] * 100:.2f}%)")
    return result
