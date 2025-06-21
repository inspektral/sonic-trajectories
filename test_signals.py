import numpy as np
import synth
import utils

# Amplitude

def square_slow(amount=0.5):
    """
    Square slow attack: 4 2.5s 440hz square wave long notes, triangular envelope
    """
    modulator = np.tile(synth.ahd(1, 0, 1), 4)               
    square = synth.square_wave(modulator, [440], duration=10)
    audio = synth.norm(square)
    return audio, modulator

def square_fast(amount=0.5):
    """
    Square short: short burst (1200 samples) 44hz square wave, exponential envelope
    """
    modulator = np.tile(synth.adsr(0.01, 0.05, 0, 0, 10, 1), 20)
    square = synth.square_wave(modulator, [440], duration=10)
    audio = synth.norm(square)
    return audio, modulator


# Pitch

def square_vibrato(amount=0.1):
    """
    Square vibrato: continuous square wave oscillating between 330 and 550, vibrato starts at 0.2hz and rises to 1hz
    """
    modulator = synth.sine_wave([1], [0.2, 1], duration=10)
    saw = synth.square_wave([1], modulator*amount*220+440, duration=10)
    audio = synth.norm(saw)
    return audio, modulator

# Noise

def saw_noise(ratio=1):
    """
    Sawtooth over noise: 4 1.5s long 50hz sawtooth notes, with fast attack and slow decay, over white noise with half the amplitude (-6db) with respect to the sawtooth 
    """
    adsr = synth.adsr(0.1, 0.5, 0.0, 0.2, 1)
    modulator = np.tile(adsr, 4)
    saw = synth.sawtooth_wave(modulator, [50], duration=10)

    noise = synth.noise(duration=10)
    audio = synth.norm(saw+noise*ratio)
    return audio, modulator

def sines_noise(ratio=1):
    """
    Sines over noise: 4 1.5s long notes made of 11 stacked sines each one with a stable random frequency between 100hz and 5000hz, with fast attack and slow decay, over white noise with half the amplitude (-6db) with respect to the sawtooth 
    """
    adsr = synth.adsr(0.1, 0.5, 0.0, 0.2, 1)
    modulator = np.tile(adsr, 4)
    sines = synth.sine_wave(modulator, [np.random.randint(100, 5000)], duration=10)
    for i in range(10):
        sines = sines + synth.sine_wave(modulator, [np.random.randint(100, 5000)], duration=10)
    noise = synth.noise(duration=10)
    audio = sines+noise*ratio
    audio = synth.norm(audio)
    return audio, modulator

# Frequency content

def filter_saw(amount=0.5):
    """
    Filtered saw: stable 100hz saw with 12db/oct lowpass filter with cutoff frequency between 700hz and 1300hz, modulated by 1hz sine wave
    """
    saw = synth.sawtooth_wave([1], [100], duration=10)
    modulator = synth.sine_wave([1], [0.5], duration=10)*300+1000

    audio = synth.HLPfilter(saw, modulator*amount)
    audio = synth.norm(audio)
    modulator = utils.norm(modulator)

    return audio, modulator

def triangle_clip(amount=0.5):
    """
    Soft clipped triangle: stable 440hz trainge wave into a soft clipper with gain between 0db and 20db, modulated by a sine wave that starts at 0.2hz and ends at 1hz
    """
    modulator = (synth.sine_wave([.5, 1.5], [0.2, 1], duration=10)+1)*10
    triangle = synth.triangle_wave(modulator, [440], duration=10)

    audio = synth.soft_clip(triangle, modulator*amount)
    audio = synth.norm(audio)
    modulator = utils.norm(modulator)

    return audio, modulator

def fm_amplitude(amount=0.5):
    """
    FM triangle (modulator amplitude): stable 440hz triangle wave with modulated by 600hz triangle wave  whose amplitude is modulated between 0 and 0.5 by a sine wave that starts at 0.2hz and ends at 1hz
    """
    modulator = utils.norm(synth.sine_wave([1], [0.2, 1], duration=10))
    fm_mod = synth.triangle_wave([1], [600], duration=10)
    audio = synth.triangle_wave([1], fm_mod*amount*modulator*440+440, duration=10)
    audio = synth.norm(audio)
    return audio, modulator

def fm_frequency(amount=0.5):
    """
    FM triangle (modulator frequency): stable 440hz triangle wave with modulated by 400-600hz (modulated by a sine wave that starts at 0.2hz and ends at 1hz) triangle wave with amplitude 0.5.
    """
    modulator = utils.norm(synth.sine_wave([1], [0.2, 1], duration=10))
    fm_mod = synth.triangle_wave([1], 400+200*modulator, duration=10)
    audio = synth.triangle_wave([1], fm_mod*amount*440+440, duration=10)
    audio = synth.norm(audio)
    return audio, modulator

# Delay and reverb

def delay_noise(amount=0.5):
    """
    Noise with delay: short burst of noise with delay (0.1s, 0.7 feedback) with wet between 0 and 0.5, odulated by a sine wave that starts at 0.2hz and ends at 1hz
    """
    noise = synth.noise(np.tile(synth.adsr(0.01, 0.2, 0,0, 10, 1), 20), duration=10)
    modulator = utils.norm(synth.sine_wave([1], [0.2], duration=10))
    delay = synth.delay(noise, wet=modulator*amount)
    audio = synth.norm(delay)
    return audio, modulator

def delay_saw(amount=0.5):
    """
    Sawtooth with delay: short burst of 220hz sawtooth with delay (0.1s, 0.7 feedback) with wet between 0 and 0.5, odulated by a sine wave that starts at 0.2hz and ends at 1hz
    """
    saw = synth.sawtooth_wave(np.tile(synth.adsr(0.01, 0.2, 0,0, 10, 1), 20), [220], duration=10)
    modulator = utils.norm(synth.sine_wave([1], [0.2], duration=10))
    delay = synth.delay(saw, wet=modulator*amount)
    audio = synth.norm(delay)
    return audio, modulator

def reverb_noise(amount=0.5):
    """
    Noise with reverb: short burst of noise with reverb (1.5 decay, flat response) with wet between 0 and 0.5, odulated by a sine wave that starts at 0.2hz and ends at 1hz
    """
    noise = synth.noise(np.tile(synth.adsr(0.01, 0.2, 0,0, 10, 1), 20), duration=10)
    modulator = utils.norm(synth.sine_wave([1], [0.2], duration=10))*amount
    reverb = synth.reverb(noise, wet=modulator)
    audio = synth.norm(reverb)
    return audio, modulator

def reverb_saw(amount=0.5):
    """
    Sawtooth with reverb: short burst of 220hz sawtooth with reverb (1.5 decay, flat response) with wet between 0 and 0.5, odulated by a sine wave that starts at 0.2hz and ends at 1hz
    """
    saw = synth.sawtooth_wave(np.tile(synth.adsr(0.01, 0.2, 0,0, 10, 1), 20), [220], duration=10)
    modulator = utils.norm(synth.sine_wave([1], [0.2], duration=10))*amount
    reverb = synth.reverb(saw, wet=modulator)
    audio = synth.norm(reverb)
    return audio, modulator



def get_tests():
    """
    Returns a list of test functions and their names.
    """
    return {
        "square_slow": square_slow,
        "square_fast": square_fast,
        "square_vibrato": square_vibrato,
        "saw_noise": saw_noise,
        "sines_noise": sines_noise,
        "filter_saw": filter_saw,
        "triangle_clip": triangle_clip,
        "fm_amplitude": fm_amplitude,
        "fm_frequency": fm_frequency,
        "delay_noise": delay_noise,
        "delay_saw": delay_saw,
        "reverb_noise": reverb_noise,
        "reverb_saw": reverb_saw
    }