import os
import pandas as pd
import numpy as np

RAVDESS = "./train-data/ravdess/audio_speech_actors_01-24/"
CREMA = "./train-data/cremad/AudioWAV/"


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

