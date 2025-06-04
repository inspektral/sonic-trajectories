import numpy as np
import synth
import utils

# Amplitude

def square_slow(amount=0.5):
    modulator = np.tile(synth.ahd(1, 0, 1), 4)               
    square = synth.square_wave(modulator, [440], duration=10)
    audio = synth.norm(square)
    return audio, modulator

def square_fast(amount=0.5):
    modulator = np.tile(synth.adsr(0.01, 0.05, 0, 0, 10, 1), 20)
    square = synth.square_wave(modulator, [440], duration=10)
    audio = synth.norm(square)
    return audio, modulator


# Pitch

def saw_vibrato(amount=0.1):
    modulator = synth.sine_wave([1], [0.2, 1], duration=10)
    saw = synth.square_wave([1], modulator*amount*220+220, duration=10)
    audio = synth.norm(saw)
    return audio, modulator

# Noise

def saw_noise(ratio=0.5):
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

# Frequency content

def filter_saw(amount=0.5):
    saw = synth.sawtooth_wave([1], [100], duration=10)
    modulator = synth.sine_wave([1], [0.5], duration=10)*300+1000

    audio = synth.HLPfilter(saw, modulator*amount)
    audio = synth.norm(audio)
    modulator = utils.norm(modulator)

    return audio, modulator

def triangle_clip(amount=0.5):
    modulator = (synth.sine_wave([.5, 1.5], [0.2, 1], duration=10)+1)*10
    triangle = synth.triangle_wave(modulator, [440], duration=10)

    audio = synth.soft_clip(triangle, modulator*amount)
    audio = synth.norm(audio)
    modulator = utils.norm(modulator)

    return audio, modulator

def fm_amplitude(amount=0.5):
    modulator = utils.norm(synth.sine_wave([1], [0.2, 1], duration=10))
    fm_mod = synth.triangle_wave([1], [600], duration=10)
    audio = synth.triangle_wave([1], fm_mod*amount*modulator*440+440, duration=10)
    audio = synth.norm(audio)
    return audio, modulator

def fm_frequency(amount=0.5):
    modulator = utils.norm(synth.sine_wave([1], [0.2, 1], duration=10))
    fm_mod = synth.triangle_wave([1], 400+200*modulator, duration=10)
    audio = synth.triangle_wave([1], fm_mod*amount*440+440, duration=10)
    audio = synth.norm(audio)
    return audio, modulator

# Delay and reverb

def delay_noise(amount=0.5):
    noise = synth.noise(np.tile(synth.adsr(0.01, 0.2, 0,0, 10, 1), 20), duration=10)
    modulator = utils.norm(synth.sine_wave([1], [0.2], duration=10))
    delay = synth.delay(noise, wet=modulator*amount)
    audio = synth.norm(delay)
    return audio, modulator

def delay_saw(amount=0.5):
    saw = synth.sawtooth_wave(np.tile(synth.adsr(0.01, 0.2, 0,0, 10, 1), 20), [220], duration=10)
    modulator = utils.norm(synth.sine_wave([1], [0.2], duration=10))
    delay = synth.delay(saw, wet=modulator*amount)
    audio = synth.norm(delay)
    return audio, modulator

def reverb_noise(amount=0.5):
    noise = synth.noise(np.tile(synth.adsr(0.01, 0.2, 0,0, 10, 1), 20), duration=10)
    modulator = utils.norm(synth.sine_wave([1], [0.2], duration=10))*amount
    reverb = synth.reverb(noise, wet=modulator)
    audio = synth.norm(reverb)
    return audio, modulator

def reverb_saw(amount=0.5):
    saw = synth.sawtooth_wave(np.tile(synth.adsr(0.01, 0.2, 0,0, 10, 1), 20), [220], duration=10)
    modulator = utils.norm(synth.sine_wave([1], [0.2], duration=10))*amount
    reverb = synth.reverb(saw, wet=modulator)
    audio = synth.norm(reverb)
    return audio, modulator

