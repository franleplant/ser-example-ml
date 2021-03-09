import sys
import librosa
import pandas as pd
import numpy as np


from get_data import normalize_ravdess, normalize_crema
from augment_data import augment_data
from get_features import get_features


def get_training_data(dataframe):

    # print(ravdess.head())
    print(dataframe.head())
    X = []
    Y = []

    for index, row in dataframe.iterrows():

        # print(index, row)
        emotion = row["emotion"]
        path = row["path"]
        print("processing file", path)
        # print(emotion, path)
        # duration and offset are used to take care of the no audio
        # in start and the ending of each audio files as seen above.
        data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

        # augment_data returns the original data as the first element
        for augmented_data in augment_data(data, sample_rate):
            features = get_features(data, sample_rate)
            # Storing a single list of all the features plus the last one is the target label
            X.append(features)
            Y.append(emotion)

    df = pd.DataFrame(X)
    df["label"] = Y

    return df


def get_training_data_ravdess():
    print("processing ravdess")
    ravdess = normalize_ravdess()
    df = get_training_data(ravdess)
    # Save it as a cache
    df.to_csv("training_data_ravdess.csv")

def get_training_data_crema():
    print("processing crema")
    crema = normalize_crema()
    df = get_training_data(crema)
    # Save it as a cache
    df.to_csv("training_data_crema.csv")



def get_training_data_all():
    # TODO paralelize these funcs
    get_training_data_crema()
    get_training_data_ravdess()

if __name__ == "__main__":
    [script_name, source_id] = sys.argv
    if not source_id:
        print("get all training data, you should probably paralelize individually")
        get_training_data_all()
    else:
        if source_id == "ravdess":
            get_training_data_ravdess()
        if source_id == "crema":
            get_training_data_crema()
        else:
            print("invalid dataset id")

