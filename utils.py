from IPython.display import Audio, display
import numpy as np
import librosa
import matplotlib.pyplot as plt

def add_padding(audio_array, sr=44100):
    """
    Add padding to the audio array to make it a multiple of the sample rate.
    """
    audio_padded = np.zeros((audio_array.size+sr))
    audio_padded[int(sr/2):int(sr/2)+audio_array.size] = audio_array
    return  audio_padded

def spectrogram(audio, sr=44100, title='Spectrogram'):
    
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=8192)), ref=np.max), y_axis='log', x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()
