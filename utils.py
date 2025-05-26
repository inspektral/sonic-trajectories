from IPython.display import Audio, display
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

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

def distances(repr:np.ndarray, top_n:int=0):
    distances = np.power(repr[:, 1:] - repr[:, :-1], 2)
    print(f"distances shape: {distances.shape}")

    if top_n > 0:
        spans = np.max(distances, axis=1) - np.min(distances, axis=1)
        max_indices = np.argsort(spans)[-top_n:]
        print(f"max_indices: {max_indices}")
        distances = distances[max_indices, :]
    
    distances = np.sum(distances, axis=0)
    distances = np.sqrt(distances)
    return distances

def cosine_similarity(repr: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between consecutive columns.
    
    Parameters:
        repr (np.ndarray): Feature matrix of shape (num_features, num_frames)
    
    Returns:
        np.ndarray: Array of cosine similarities between consecutive frames
    """
    # Get consecutive column pairs
    a = repr[:, 1:]  # all columns except first
    b = repr[:, :-1] # all columns except last
    
    # Compute dot products for all pairs
    dot_products = np.sum(a * b, axis=0)
    
    # Compute norms for all columns
    norms_a = np.linalg.norm(a, axis=0)
    norms_b = np.linalg.norm(b, axis=0)
    
    # Avoid division by zero with small epsilon
    eps = 1e-8
    similarities = dot_products / (norms_a * norms_b + eps)
    
    return similarities

def norm(arr):
    return (arr-np.min(arr))/ np.max(arr-np.min(arr))

def stretch_array(arr:np.ndarray, target_length:int):

    """
    Stretch/compress array to target length using linear interpolation.
    
    Parameters:
    source (np.ndarray): Source array to be stretched/compressed
    target_length (int): Desired length of output array
    
    Returns:
    np.ndarray: Interpolated array of length target_length
    """
    old_indices = np.arange(len(arr))
    new_indices = np.linspace(0, len(arr) - 1, target_length)
    
    return np.interp(new_indices, old_indices, arr)


def smoothing(arr:np.ndarray, window_size:int=5):
    """
    Apply a moving average filter to smooth the array.
    """
    if window_size % 2 == 0:
        window_size += 1
    half_window = window_size // 2
    padded_arr = np.pad(arr, (half_window, half_window), mode='edge')
    smoothed_arr = np.convolve(padded_arr, np.ones(window_size)/window_size, mode='valid')
    return smoothed_arr

def window(length:int):
    window = np.zeros(length)
    window[1:-1] = 0.5
    window[2:-2] = 1

    return window

def plot_with_derivatives(x, dx, ddx, modulator,  title='Plot', label='label'):
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    single_plot(x, modulator, label=label)
    
    plt.subplot(3, 1, 2)
    single_plot(dx, modulator, label='First Derivative')

    plt.subplot(3, 1, 3)
    single_plot(ddx, modulator, label='Second Derivative')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def single_plot(x, modulator, label='label'):
    plt.plot(norm(x)*window(len(x)), label=label)
    plt.plot(norm(stretch_array(modulator, len(x)))*window(len(x)), alpha=0.8, label="modulator")
    plt.legend()

def plot_heatmap(arr, title='Heatmap'):
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(arr, cmap='viridis')
    ax.invert_yaxis()
    plt.title(title)
    plt.tight_layout()
    plt.show()
