import numpy as np
import librosa


def calc_zero_crossing_rate(data):
    return np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)


def calc_chroma_stft(data, sample_rate):
    stft = np.abs(librosa.stft(data))
    return np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)


def calc_mfcc(data, sample_rate):
    return np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)


def calc_rms(data):
    return np.mean(librosa.feature.rms(y=data).T, axis=0)


def calc_melspectrogram(data, sample_rate):
    return np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)


def get_features(data, sample_rate):
    zcr = calc_zero_crossing_rate(data)
    chroma = calc_chroma_stft(data, sample_rate)
    mfcc = calc_mfcc(data, sample_rate)
    rms = calc_rms(data)
    mel = calc_melspectrogram(data, sample_rate)

    # print("features")
    # print("zcr", len(zcr))
    # print("chroma", len(chroma))
    # print("mfcc", len(mfcc))
    # print("rms", len(rms))
    # print("mel", len(mel))

    # we unfold all values into a very long array, instead of having an array of arrays
    return np.hstack([zcr, chroma, mfcc, rms, mel])


# def run(path):
# duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
# data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

# without augmentation
# res1 = extract_features(data)
# result = np.array(res1)

# data with noise
# noise_data = noise(data)
# res2 = extract_features(noise_data)
# result = np.vstack((result, res2)) # stacking vertically

# data with stretching and pitching
# new_data = stretch(data)
# data_stretch_pitch = pitch(new_data, sample_rate)
# res3 = extract_features(data_stretch_pitch)
# result = np.vstack((result, res3)) # stacking vertically

# return result
