import numpy as np
import pandas as pd
from matplotlib import pyplot
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import librosa
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from get_features import get_features


print(tf.__version__)
print(keras.__version__)

labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
model = keras.models.load_model("./model2.h5")
model.summary()


validation_audio = [
     "./happy.mp3",
     "./happy.m4a",
     "./happy2.m4a",
     "./lucas_nienpedo.mp3",
     "./sol_maichu.m4a",
     "./fran_scared.m4a",
     "./fran_scared2.m4a",
    "./lucas_cursing.mp3",
    "./nacho_happy.mp3",
]

for path in validation_audio:
    print("procesing", path)
    data, sampling_rate = librosa.load(path, duration=2.5, offset=0.6)
    features = get_features(data, sampling_rate)
    # features_transposed = np.expand_dims([features], axis=2)
    # print(features.shape())
    f = np.array(features)
    print(f.shape)
    f = f.reshape(162)
    print(f.shape)

    # THIS MOTHER FUCKER REQUIRES THIS AMOUNT OF WORKAROUND
    # TO PREDICT A SINGLE FUCKING INSTANCE IM GOING INSANE
    res = model.predict(np.array([features,]))
    # res = model(f)
    max_id = np.argmax(res[0])
    print("prediction", labels[max_id])
    print("predictions")
    for score, label in zip(res[0], labels):
        print(label, score)
