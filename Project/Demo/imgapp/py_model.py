from pathlib import Path
import tensorflow as tf
from keras.applications import InceptionV3
from keras.applications import MobileNetV3Large

model = InceptionV3(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(224, 224, 3),
    pooling=None,
    classes=1000,
    classifier_activation="softmax",


)

output = Path.cwd() / "saved_models" / "InceptionV3"
model.save(output)
