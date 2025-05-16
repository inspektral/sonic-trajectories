import dac
import torch
from audiotools import AudioSignal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DACSound:

    def __init__(self, model, audio, sr):
        self.audio = audio
        self.sr = sr
        self.model = model

        snd = AudioSignal(audio, sr)
        snd.to(model.device)
        snd_x = model.preprocess(snd.audio_data, snd.sample_rate)

        with torch.no_grad():
            self.z, self.codes, self.latents, _, _ = model.encode(snd_x)

    def decode(self):
        with torch.no_grad():
            snd_decoded = self.model.decode(self.z)
        audio = snd_decoded.cpu().detach().numpy().flatten()
        return audio
    
    def decode_from_latents(self):
        with torch.no_grad():
            snd_z,_a,_b = self.model.quantizer.from_latents(self.latents)
            snd_decoded = self.model.decode(snd_z)
        audio = snd_decoded.cpu().detach().numpy().flatten()
        return audio



    

    