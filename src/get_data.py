import os
import pandas as pd
import numpy as np

RAVDESS = "./train-data/ravdess/audio_speech_actors_01-24/"


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
        actor = os.listdir(RAVDESS + dir)
        for file in actor:
            name, extension = file.split(".")
            [
                modality,
                channel,
                emotion,
                intensity,
                statement,
                repetition,
                actor,
            ] = name.split("-")
            emotion_label = emotion_map[int(emotion)]
            emotion_path = RAVDESS + dir + "/" + file
            rows.append((emotion_label, emotion_path))

    df = pd.DataFrame(rows, columns=["emotion", "path"])
    # print(df.head())

    return df
