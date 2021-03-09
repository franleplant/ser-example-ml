import numpy as np
import librosa


def noise(data):
    """Add white noice to the librosa sound """
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data


def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)


def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)


def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

def augment_data(data, sample_rate):
    """
    grab a single audio file and by augmenting it turn it into 4,
    we return a list made of the original data and the new data
    """

    return [
        data,
        noise(data),
        pitch(stretch(data), sample_rate),
        shift(data),
    ]

