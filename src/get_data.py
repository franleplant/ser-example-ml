import os
import pandas as pd
import numpy as np

RAVDESS = "./train-data/ravdess/audio_speech_actors_01-24/"
CREMA = "./train-data/cremad/AudioWAV/"
SAVEE = "./train-data/savee/"
TESS = "./train-data/tess/TESS Toronto emotional speech set data/"


def normalize_ravdess():
    """
    grab the files and turn them into a pandas data frame
    of the shape emotion: "angry", path: "path to the file"
    """
    emotion_map = {
        1: "neutral",
        2: "calm",
        3: "happy",
        4: "sad",
        5: "angry",
        6: "fear",
        7: "disgust",
        8: "surprise",
    }
    rows = []
    for dir in os.listdir(RAVDESS):
        # as their are 20 different actors in our previous directory we need to extract files for each actor.
        actor_dir = os.listdir(RAVDESS + dir)
        for file in actor_dir:
            if file.startswith("."):
                print("skipping dotfile", file)
                continue
            name, extension = file.split(".")
            [
                modality,
                channel,
                emotion_id,
                intensity,
                statement,
                repetition,
                actor,
            ] = name.split("-")
            emotion = emotion_map[int(emotion_id)]
            file_path = RAVDESS + dir + "/" + file
            rows.append((emotion, file_path))

    df = pd.DataFrame(rows, columns=["emotion", "path"])
    # print(df.head())

    return df


def normalize_crema():
    emotion_map = {
        'NEU': "neutral",
        'HAP': "happy",
        'SAD': "sad",
        'ANG': "angry",
        'FEA': "fear",
        'DIS': "disgust",
    }

    rows = []

    for file in os.listdir(CREMA) :
        if file.startswith("."):
            print("skipping dotfile", file)
            continue

        # storing file paths
        file_path = CREMA + file

        # storing file emotions
        [id, actor, emotion_id, modifier]=file.split('_')
        emotion = emotion_map[emotion_id]
        if not emotion:
            print("skipping unknown emotion file", file)
            continue

        rows.append((emotion, file_path))

    df = pd.DataFrame(rows, columns=["emotion", "path"])
    # print(df.head())
    return df

def normalize_savee():
    rows = []

    emotion_map = {
        'n': "neutral",
        'h': "happy",
        'sa': "sad",
        'a': "angry",
        'f': "fear",
        'd': "disgust",
        "su": "surprise"
    }

    for file in os.listdir(SAVEE):
        file_path = SAVEE + file
        part = file.split('_')[1]
        ele = part[:-6]
        if not ele:
            continue
        emotion = emotion_map[ele]

        rows.append((emotion, file_path))

    df = pd.DataFrame(rows, columns=["emotion", "path"])
    # print(df.head())
    return df

def normalize_tess():
    rows = []

    for directory in os.listdir(TESS):
        if directory.startswith("."):
            print("skipping dotfile", directory)
            continue
        for file in os.listdir(TESS + directory):
            if file.startswith("."):
                print("skipping dotfile", file)
                continue
            file_path = TESS + directory + '/' + file

            [name, extension] = file.split('.')
            [actor, word, emotion_id] = name.split('_')
            emotion = emotion_id
            if emotion_id=='ps':
                emotion = 'surprise'

            rows.append((emotion, file_path))

    df = pd.DataFrame(rows, columns=["emotion", "path"])
    # print(df.head())
    return df
