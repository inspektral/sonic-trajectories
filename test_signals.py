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

    sine = synth.sine_wave(np.tile(adsr, 4), [440], duration=10)

    random_sines = random_sines*ratio + sine
    random_sines = utils.norm(random_sines)
    return random_sines

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

    adsr = synth.adsr(0.1, 0.5, 0.0, 0.2, 20)
    saw = synth.sawtooth_wave(np.tile(adsr, 4), [50], duration=10)

    noise = synth.noise(duration=10)

    return saw+noise*ratio