import numpy as np
from scipy.interpolate import interp1d
print("imports done")

class WavetableSynth:
    def __init__(self, sr=44100, duration=10.0):
        self.sr = sr
        self.duration = duration
        self.samples = int(sr * duration)
        
        self.wavetable = self.gen_wavetable()

        self.base_freq = 440.0
        self.amp = 1.0

        self.mod_freq_ratio = 1.0
        self.mod_shape = 0.0
        
        self.carr_shape = 0.0
        self.fm_amount = 0.0
    

    def render(self):
        mod_audio = self.wavetable_osc(
            frequency=np.array([self.base_freq * self.mod_freq_ratio]),
            shape=np.array([self.mod_shape]),
            wavetable=self.wavetable
        )
        print(f"mod_audio shape: {mod_audio.shape}, samples: {self.samples}")
        carr_audio = self.wavetable_osc(
            frequency=mod_audio * self.fm_amount * self.base_freq + self.base_freq,
            shape=np.array([self.carr_shape]),
            wavetable=self.wavetable
        )
        return self.amp * carr_audio
    

    def gen_wavetable(self, samples = 2048, blend_values = 512):

        sine = np.sin(np.linspace(0, 2 * np.pi, samples))
        triangular =  np.roll(np.concatenate((np.linspace(-1, 1, int(samples/2)), np.linspace(1, -1, int(samples/2)))), int(-samples/4))
        square = np.sign(sine)
        sawtooth = np.linspace(1, -1, samples)

        keys = np.array([sine, triangular, sawtooth, square]) 
        positions = [0, 0.33, 0.66, 1.0]

        interp = interp1d(positions, keys, axis=0)

        blend_positions = np.linspace(0, 1, blend_values)
        wavetable = interp(blend_positions)

        return wavetable
    

    def wavetable_osc(self, frequency, shape, wavetable):
        frequencies = self.stretch_array(frequency, self.samples)
        shapes = self.stretch_array(shape, self.samples)
        
        phase = np.cumsum(frequencies) / self.sr
        
        wave_indices_float = shapes * (wavetable.shape[0] - 1)
        wave_idx1 = np.clip(wave_indices_float.astype(int), 0, wavetable.shape[0] - 1)
        wave_idx2 = np.clip(wave_idx1 + 1, 0, wavetable.shape[0] - 1)
        wave_blend = wave_indices_float - wave_idx1
        
        table_positions = ((phase % 1.0) * wavetable.shape[1]).astype(int) % wavetable.shape[1]
        
        sample1 = wavetable[wave_idx1, table_positions]
        sample2 = wavetable[wave_idx2, table_positions]
        output = sample1 * (1 - wave_blend) + sample2 * wave_blend
        
        return output
    
    
    def stretch_array(self, arr:np.ndarray, target_length:int):
        old_indices = np.arange(len(arr))
        new_indices = np.linspace(0, len(arr) - 1, target_length)
        
        return np.interp(new_indices, old_indices, arr)
        
if __name__ == "__main__":
    synth = WavetableSynth()
    audio = synth.render()
    print(audio)