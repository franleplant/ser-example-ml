import numpy as np
from matplotlib import pyplot
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import librosa
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from get_features import get_features


print(tf.__version__)
print(keras.__version__)



# data, sampling_rate = librosa.load("./happy.mp3")
# data, sampling_rate = librosa.load("./happy.m4a")
data, sampling_rate = librosa.load("./happy2.m4a")
features = get_features(data, sampling_rate)
features_transposed = np.expand_dims([features], axis=2)

# print("input features")
# print(len(features_transposed))
# print(features_transposed)

model = keras.models.load_model("./model.h5")
# model.summary()

labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']

res = model.predict(features_transposed)

max_id = np.argmax(res[0])

print("prediction", labels[max_id])
print("predictions")
for score, label in zip(res[0], labels):
    print(label, score)
