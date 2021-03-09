import librosa
import pandas as pd
import numpy as np


from get_data import normalize_ravdess
from get_features import get_features


def get_training_data():
    ravdess = normalize_ravdess()

    # print(ravdess.head())

    X = []
    Y = []

    for index, row in ravdess.iterrows():

        # print(index, row)
        emotion = row["emotion"]
        path = row["path"]
        # print(emotion, path)
        # duration and offset are used to take care of the no audio
        # in start and the ending of each audio files as seen above.
        data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
        features = get_features(data, sample_rate)
        # Storing a single list of all the features plus the last one is the target label
        X.append(features)
        Y.append(emotion)

    df = pd.DataFrame(X)
    df["label"] = Y

    return df


df = get_training_data()
# Save it as a cache
df.to_csv("training_data.csv")

print(df.head())
