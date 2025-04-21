from IPython.display import Audio, display
import numpy as np

def display(audio_array, sr=44100):
    """
    Displays an audio file in a Jupyter notebook.
    
    Parameters:
    audio (numpy.ndarray): Audio data to be displayed.
    sr (int): Sample rate of the audio data.
    """
    audio_padded = np.zeros((len(audio_array)+sr))
    audio_padded[int(sr/2):int(sr/2)+len(audio_array)] = audio_array

    display(Audio(audio_padded, rate=sr))
