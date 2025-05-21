import numpy as np
import synth
import utils

def random_sines(ratio:float=0.5):
    """
    Generate test signal mixing random sine waves with reference tone.
    
    Generates complex signal by:
    1. Summing 10 random sine waves with random frequencies (100-1000 Hz)
    2. Adding reference 440 Hz sine wave with ADSR envelope
    3. Mixing signals according to ratio parameter
    4. Normalizing output
    
    Parameters:
        ratio (float): Mix ratio between random sines (ratio) and reference tone (1-ratio). 
                      Default 0.5 for equal mix.
    
    Returns:
        np.ndarray: Normalized audio signal of 10 seconds duration
    
    Example:
        >>> signal = random_sines(ratio=0.7)  # 70% random, 30% reference
    """
    adsr = synth.adsr(0.1, 0.5, 0.0, 0.2, 20)
    random_sines = synth.sine_wave(np.random.rand(10), np.random.randint(100, 1000, 10), duration=10)
    for i in range(10):
        random_sines = random_sines + synth.sine_wave(np.random.rand(10), np.random.randint(100, 1000, 10), duration=10)

    modulator = np.tile(adsr, 4)
    sine = synth.sine_wave(modulator, [440], duration=10)

    audio = random_sines*ratio + sine
    audio = synth.norm(audio)

    return audio, modulator

def saw_noise(ratio=0.5):
    """
    Generate test signal mixing sawtooth wave with noise.

    Generates complex signal by:
    1. Generating sawtooth wave with ADSR envelope
    2. Generating white noise
    3. Mixing signals according to ratio parameter
    4. Normalizing output

    Parameters:
        ratio (float): Mix ratio between sawtooth wave (ratio) and noise (1-ratio). 
                      Default 0.5 for equal mix.

    Returns:
        np.ndarray: Normalized audio signal of 10 seconds duration

    Example:
        >>> signal = saw_noise(ratio=0.7)  # 70% sawtooth, 30% noise
    """

    adsr = synth.adsr(0.1, 0.5, 0.0, 0.2, 1)
    modulator = np.tile(adsr, 4)
    saw = synth.sawtooth_wave(modulator, [50], duration=10)

    noise = synth.noise(duration=10)
    audio = synth.norm(saw+noise*ratio)

    return audio, modulator

def sines_noise(ratio=0.5):
    adsr = synth.adsr(0.1, 0.5, 0.0, 0.2, 1)
    modulator = np.tile(adsr, 4)
    sines = synth.sine_wave(modulator, [np.random.randint(100, 5000)], duration=10)
    for i in range(10):
        sines = sines + synth.sine_wave(modulator, [np.random.randint(100, 5000)], duration=10)
    noise = synth.noise(duration=10)
    audio = sines+noise*ratio
    audio = synth.norm(audio)
    return audio, modulator

def filter_saw(amount=0.5):
    saw = synth.sawtooth_wave([1], [100], duration=10)
    modulator = synth.sine_wave([1], [0.5], duration=10)*300+1000

    audio = synth.HLPfilter(saw, modulator*amount)
    audio = synth.norm(audio)
    modulator = utils.norm(modulator)

    return audio, modulator

def triangle_clip(amount=0.5):
    modulator = (synth.sine_wave([1, 3], [0.2, 1], duration=10)+1)*10
    triangle = synth.triangle_wave(modulator, [440], duration=10)

    audio = synth.soft_clip(triangle, modulator*amount)
    audio = synth.norm(audio)
    modulator = utils.norm(modulator)

    return audio, modulator

