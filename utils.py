from IPython.display import Audio, display
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

# music2latent
from music2latent.music2latent.inference import EncoderDecoder

# DAC
from audiotools import AudioSignal
import torch
import dacsound
import dac

# CORRELATION
from scipy.signal import correlate, correlation_lags

# OPTIMIZATIONS
import scipy.signal
from skopt import gp_minimize
from skopt.space import Integer
from scipy.optimize import minimize





def add_padding(audio_array, sr=44100):
    """
    Add padding to the audio array to make it a multiple of the sample rate.
    """
    audio_padded = np.zeros((audio_array.size+sr))
    audio_padded[int(sr/2):int(sr/2)+audio_array.size] = audio_array
    return  audio_padded

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

def norm_stretch(arr, length):
    arr = norm(arr)
    arr = stretch_array(arr, length)
    return arr


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

def plot_with_derivatives(x, dx, ddx, modulator,  title='Plot', label='label', small=False):
    if small:
        plt.figure(figsize=(6, 4))
    else:
        plt.figure(figsize=(10, 6))
    
    plt.subplot(3, 1, 1)
    if isinstance(x, dict):
        multiplot(x, modulator, label=label)
    else:
        single_plot(x, modulator, label=label)
    
    plt.subplot(3, 1, 2)
    if isinstance(dx, dict):
        multiplot(dx, modulator, label='First Derivative')
    else:
        single_plot(dx, modulator, label='First Derivative')

    plt.subplot(3, 1, 3)
    if isinstance(ddx, dict):
        multiplot(ddx, modulator, label='Second Derivative')
    else:
        single_plot(ddx, modulator, label='Second Derivative')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def single_plot(x, modulator, label='label'):
    plt.plot(norm(x)*window(len(x)), label=label)
    plt.plot(norm(stretch_array(modulator, len(x)))*window(len(x)), alpha=0.8, label="modulator")
    plt.legend()

def multiplot(x_dict, modulator, label):
    max_length = max([len(x) for x in x_dict.values()])
    for key in x_dict:
        x = stretch_array(x_dict[key], max_length)
        x = x[10:-10]
        plt.plot(norm(x)*window(len(x)), label=f'{label} {key}')
    plt.plot(norm(stretch_array(modulator, max_length))*window(max_length), alpha=0.8, label="modulator")
    plt.legend()


def plot_heatmap(arr, title='Heatmap', small=False):
    if small:
        plt.figure(figsize=(6, 4))
    else:
        plt.figure(figsize=(10, 6))

    ax = sns.heatmap(arr, cmap='viridis')
    ax.invert_yaxis()
    plt.title(title)
    plt.tight_layout()
    plt.show()

# metrics

def magnitude(repr:np.ndarray):
    return np.linalg.norm(repr, axis=0)

def distances(repr:np.ndarray, top_n:int=0):
    distances = np.power(repr[:, 1:] - repr[:, :-1], 2)

    if top_n > 0:
        spans = np.max(distances, axis=1) - np.min(distances, axis=1)
        max_indices = np.argsort(spans)[-top_n:]
        print(f"max_indices: {max_indices}")
        distances = distances[max_indices, :]
    
    distances = np.sum(distances, axis=0)
    distances = np.sqrt(distances)
    return distances

def cosine_similarity(repr: np.ndarray) -> np.ndarray:
    a = repr[:, 1:]  # all columns except first
    b = repr[:, :-1] # all columns except last
    
    dot_products = np.sum(a * b, axis=0)
    
    norms_a = np.linalg.norm(a, axis=0)
    norms_b = np.linalg.norm(b, axis=0)
    
    eps = 1e-8
    similarities = dot_products / (norms_a * norms_b + eps)
    
    return similarities

def calc_metric(repr: np.ndarray, metric:str):
    """
    Calculate the specified metric for the given representation.
    
    Parameters:
    repr (np.ndarray): The representation array.
    metric (str): The metric to calculate ('magnitude', 'distances', 'cosine_similarity').
    
    Returns:
    np.ndarray: The calculated metric.
    """
    metrics = {
        'magnitude': magnitude,
        'distances': distances,
        'cosine_similarity': cosine_similarity
    }

    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}. Available metrics: {list(metrics.keys())}")
    
    return metrics[metric](repr)

get_available_metrics = lambda: ['magnitude', 'distances', 'cosine_similarity']

# representations

def get_mfcc(audio, sr=44100):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    return mfcc

def get_cqt(audio, sr=44100):
    spectrum = librosa.amplitude_to_db(np.abs(librosa.cqt(audio, sr=sr)), ref=np.max)
    return spectrum

def get_spectrum(audio, sr=44100):
    spectrum = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=4096)), ref=np.max)
    return spectrum


def get_music2latent(audio, hop_size=1024, sr=44100):
    encdec = EncoderDecoder()
    all_latent_windows = []


    for i in range(int(4096/hop_size)):
        to_encode = audio[int(i*hop_size):]
        latent = encdec.encode(to_encode)
        latent = latent.cpu().numpy()[0,:,:]
        all_latent_windows.append(latent)

    min_length = min([l.shape[1] for l in all_latent_windows])
    all_latent_windows = [l[:,:min_length] for l in all_latent_windows]

    latent = np.stack(all_latent_windows, axis=2).reshape(all_latent_windows[0].shape[0], -1)

    return latent

def get_dac(audio, sr=44100):
    device = torch.device('cpu') # or 'cuda'

    model_path = dac.utils.download(model_type="44khz")
    model = dac.DAC.load(model_path)
    model.to(device);
    model.eval();

    dac_audio = dacsound.DACSound(model, audio, sr)
    latents = dac_audio.latents.cpu().numpy()[0,:,:]

    return latents


representations = {
    'mfcc': get_mfcc,
    'cqt': get_cqt,
    'spectrum': get_spectrum,
    'music2latent': get_music2latent,
    'dac': get_dac
}

def get_representations(audio, sr=44100):

    reprs = {}
    for name, func in representations.items():
        print(f"Computing {name} representation...")
        reprs[name] = func(audio)
    
    return reprs

def calc_representation(audio:np.ndarray, repr:str, sr=44100):
    """
    Calculate the specified representation for the given audio.
    
    Parameters:
    audio (np.ndarray): The audio signal.
    repr (str): The representation to calculate ('mfcc', 'cqt', 'spectrum', 'music2latent', 'dac').
    sr (int): Sample rate of the audio signal.
    
    Returns:
    np.ndarray: The calculated representation.
    """
    representations = {
        'mfcc': get_mfcc,
        'cqt': get_cqt,
        'spectrum': get_spectrum,
        'music2latent': get_music2latent,
        'dac': get_dac
    }

    if repr not in representations:
        raise ValueError(f"Unknown representation: {repr}. Available representations: {list(representations.keys())}")
    
    return representations[repr](audio, sr=sr)

get_available_representations = lambda: list(representations.keys())

# Batch metrics calculation

def calculate_metric(representations, func, parameters=None):
    metrics = {}
    d_metrics = {}
    dd_metrics = {}
    for key in representations.keys():
        metric = func(representations[key], **(parameters or {}))
        metrics[key] = metric
        d_metric = np.diff(metric)
        d_metrics[key] = d_metric
        dd_metric = np.diff(d_metric)
        dd_metrics[key] = dd_metric
    return metrics, d_metrics, dd_metrics

# CORRELATION

def get_correlations(repr:dict, modulator:np.ndarray) -> dict:
    correlations = {}
    for key in repr.keys():
        x = repr[key]
        correlations[key] = calc_correlation(modulator, x)
    return correlations

def calc_correlation(modulator, x):
    x = stretch_array(x, len(modulator))

    y = modulator[10:-10]
    y = norm(y)

    x = x[10:-10]
    x = norm(x)
    
    return np.corrcoef(x, y)[0, 1]

import numpy as np

# smooth

def smooth(sig: np.ndarray, smoothing: float | int) -> np.ndarray:
    window_size = max(1, int(round(smoothing)))
    
    window = np.ones(window_size) / window_size

    pad_width = window_size // 2
    padded_sig = np.pad(sig, pad_width, mode='edge')
    
    smoothed = scipy.signal.fftconvolve(padded_sig, window, mode='valid')

    # Ensure the output length matches the input length
    if len(smoothed) < len(sig):
        smoothed = np.pad(smoothed, (0, len(sig) - len(smoothed)), mode='edge')
    elif len(smoothed) > len(sig):
        smoothed = smoothed[:len(sig)]
    
    return smoothed

# Optimizations

def calc_best_smoothing(sig, modulator):
    
    def smooth_objective(smoothing_factor):
        smoothed = smooth(sig, smoothing_factor[0])
        correlation = np.abs(np.corrcoef(smoothed, modulator)[0, 1])
        return -correlation  # Minimize negative = maximize positive
    
    result = gp_minimize(smooth_objective, 
                        dimensions=[Integer(1, 5000)],
                        n_calls=50,  # Only 50 evaluations!
                        random_state=42)
    
    return result.x[0]


def calc_best_exponent(sig, modulator):

    def exponent_objective(exponent):
        transformed = norm(sig ** exponent[0])
        correlation = np.abs(np.corrcoef(transformed, modulator)[0, 1])
        return -correlation

    result = minimize(exponent_objective, x0=[1.0], bounds=[(0.001, 10)])
    best_exponent = result.x[0]
    return best_exponent