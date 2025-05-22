
import numpy as np

def noise(amp:np.ndarray=[1.0], duration=1.0, sr=44100):
    """
    Generate white noise.

    Parameters:
    duration (float): Duration of the noise in seconds.

    Returns:
    numpy.ndarray: Array of white noise samples.
    """
    num_samples = int(sr * duration)
    amplitudes = stretch_array(amp, num_samples)
    noise = np.random.normal(0, 1, num_samples)*amplitudes
    return noise

def sine_wave(amp:np.ndarray, frequency:np.ndarray, duration=1.0, sr=44100):
    """
    Generate a sine wave.

    Parameters:
    frequency (float): Frequency of the sine wave in Hz.
    duration (float): Duration of the sine wave in seconds.

    Returns:
    numpy.ndarray: Array of sine wave samples.
    """
    num_samples = int(sr * duration)
    amplitudes = stretch_array(amp, num_samples)
    frequencies = stretch_array(frequency, num_samples)
    
    phase = np.cumsum(frequencies) / sr
    sine_wave = amplitudes * np.sin(2 * np.pi * phase)
    
    return sine_wave

def square_wave(amp:np.ndarray, frequency:np.ndarray, duration=1.0, sr=44100):
    """
    Generate a square wave.

    Parameters:
    frequency (float): Frequency of the square wave in Hz.
    duration (float): Duration of the square wave in seconds.

    Returns:
    numpy.ndarray: Array of square wave samples.
    """
    num_samples = int(sr * duration)
    amplitudes = stretch_array(amp, num_samples)
    frequencies = stretch_array(frequency, num_samples)
    
    phase = np.cumsum(frequencies) / sr
    square_wave = amplitudes * np.sign(np.sin(2 * np.pi * phase))
    
    return square_wave

def sawtooth_wave(amp:np.ndarray, frequency:np.ndarray, duration=1.0, sr=44100):
    """
    Generate a sawtooth wave.

    Parameters:
    frequency (float): Frequency of the sawtooth wave in Hz.
    duration (float): Duration of the sawtooth wave in seconds.

    Returns:
    numpy.ndarray: Array of sawtooth wave samples.
    """
    num_samples = int(sr * duration)
    amplitudes = stretch_array(amp, num_samples)
    frequencies = stretch_array(frequency, num_samples)
    
    phase = np.cumsum(frequencies) / sr
    sawtooth_wave = amplitudes * (2 * (phase - np.floor(phase + 0.5)))
    
    return sawtooth_wave

def triangle_wave(amp:np.ndarray, frequency:np.ndarray, duration=1.0, sr=44100):
    """
    Generate a triangle wave.

    Parameters:
    frequency (float): Frequency of the triangle wave in Hz.
    duration (float): Duration of the triangle wave in seconds.

    Returns:
    numpy.ndarray: Array of triangle wave samples.
    """
    num_samples = int(sr * duration)
    amplitudes = stretch_array(amp, num_samples)
    frequencies = stretch_array(frequency, num_samples)
    
    phase = np.cumsum(frequencies) / sr
    triangle_wave = amplitudes * (2 * np.abs(2 * (phase - np.floor(phase + 0.5))) - 1)
    
    return triangle_wave
    

def adsr(attack:float, decay:float, sustain:float, release:float, exp:float=1.0, duration=1.0, sr=44100):
    """
    Generate an ADSR envelope.

    Parameters:
    attack (float): Attack time in seconds.
    decay (float): Decay time in seconds.
    sustain (float): Sustain level (0 to 1).
    release (float): Release time in seconds.
    exp (float): Exponential curve factor.
    duration (float): Duration of the envelope in seconds.

    Returns:
    numpy.ndarray: Array of ADSR envelope samples.
    """
    num_samples = int(sr * duration)
    attack_samples = int(sr * attack)
    decay_samples = int(sr * decay)
    release_samples = int(sr * release)

    envelope = np.zeros(num_samples)
    
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples) ** exp
    
    if decay_samples > 0:
        envelope[attack_samples:attack_samples + decay_samples] = \
            1 - (1 - sustain) * np.linspace(0, 1, decay_samples) ** exp
    
    if sustain > 0:
        envelope[attack_samples + decay_samples:num_samples - release_samples] = sustain
    
    if release_samples > 0:
        envelope[-release_samples:] = sustain * (1 - np.linspace(0, 1, release_samples) ** exp)
    
    return envelope

def HLPfilter(audio:np.ndarray, cutoff_cv:np.ndarray, highpass=False, sr=44100) -> np.ndarray:
    """
    Filter audio using a 12 dB/oct (first-order) IIR filter with a modulated cutoff.

    Parameters:
        audio (np.ndarray): Input audio signal (1D)
        cutoff_cv (np.ndarray): Cutoff frequency control signal (1D, Hz)
        sample_rate (int): Sample rate in Hz
        highpass (bool): Whether to apply a highpass (True) or lowpass (False)

    Returns:
        np.ndarray: Filtered audio
    """
    output = np.zeros_like(audio)
    y = 0.0

    cutoff_cv = stretch_array(cutoff_cv, len(audio))

    for i in range(len(audio)):
        cutoff = cutoff_cv[i]
        x = 2 * np.pi * cutoff / sr
        alpha = x / (x + 1)

        if highpass:
            y = alpha * (y + audio[i] - (audio[i-1] if i > 0 else 0))
        else:
            y = y + alpha * (audio[i] - y)

        output[i] = y

    return output

def soft_clip(x:np.ndarray, gain:np.ndarray) -> np.ndarray:
    """
    Apply soft clipping to an audio signal.

    Parameters:
        x (np.ndarray): Input audio signal (1D)
        gain (np.ndarray): Gain control signal (1D)

    Returns:
        np.ndarray: Soft-clipped audio
    """
    gain = stretch_array(gain, len(x))

    return np.tanh(x * gain) / np.tanh(gain)


def reverb(audio, wet:np.ndarray=[1], decay= 1.5):
    """
    Apply reverb to an audio signal.

    Parameters:
        x (np.ndarray): Input audio signal (1D)
        decay (float): Decay time in seconds
    Returns:
        np.ndarray: Reverberated audio
    """
    wet = stretch_array(wet, len(audio))

    ir = noise(np.logspace(1,0,300)/10, decay)
    ir = HLPfilter(ir, [2000])
    x = np.convolve(audio, ir, mode='full')
    x = x[:len(audio)]

    x = norm(x)
    audio = norm(audio)

    x = x*wet + audio * (1 - wet)

    return x

def delay(audio, wet:np.ndarray=[1], delay_time=0.1, feedback=0.7):
    """
    Apply delay to an audio signal.

    Parameters:
        x (np.ndarray): Input audio signal (1D)
        delay_time (float): Delay time in seconds
        feedback (float): Feedback amount (0 to 1)
    Returns:
        np.ndarray: Delayed audio
    """
    wet = stretch_array(wet, len(audio))

    delay_samples = int(delay_time * 44100)
    delayed_audio = np.zeros_like(audio)
    feedback_buffer = np.zeros(delay_samples)

    for i in range(len(audio)):
        delayed_audio[i] = audio[i] + feedback_buffer[i % delay_samples]
        feedback_buffer[i % delay_samples] = audio[i] + feedback * delayed_audio[i]

    delayed_audio = delayed_audio[:len(audio)]
    delayed_audio = delayed_audio * wet
    delayed_audio = delayed_audio + audio * (1 - wet)

    return delayed_audio



# UTILS

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

def norm(x:np.ndarray, min_val:float=-1.0, max_val:float=1.0) -> np.ndarray:
    """
    Normalize an array to a specified range.

    Parameters:
        x (np.ndarray): Input array
        min_val (float): Minimum value of the output range
        max_val (float): Maximum value of the output range

    Returns:
        np.ndarray: Normalized array
    """
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x * (max_val - min_val) + min_val


