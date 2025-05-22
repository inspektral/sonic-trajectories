# Onset Detection Algorithm Test Suite

This project provides a suite of Python tools and Jupyter notebooks designed for generating and analyzing audio signals to test and evaluate onset detection algorithms. It includes various signal synthesis modules, audio effects, feature extraction utilities, and example notebooks.

## Features

### Signal Generation ([`synth.py`](synth.py))
*   Basic waveforms: [`sine_wave`](synth.py), [`square_wave`](synth.py), [`sawtooth_wave`](synth.py), [`triangle_wave`](synth.py), [`noise`](synth.py)
*   Customizable ADSR envelopes: [`adsr`](synth.py)
*   Complex pre-defined test signals in [`test_signals.py`](test_signals.py) (e.g., [`random_sines`](test_signals.py), [`filter_saw`](test_signals.py), [`delay_noise`](test_signals.py))

### Audio Effects & Processing ([`synth.py`](synth.py))
*   High/Low-pass filter: [`HLPfilter`](synth.py)
*   Soft clipping distortion: [`soft_clip`](synth.py)
*   Reverb: [`reverb`](synth.py)
*   Delay: [`delay`](synth.py)

### Analysis & Utilities ([`utils.py`](utils.py))
*   Spectrogram display: [`spectrogram`](utils.py)
*   Feature distance calculation: [`distances`](utils.py)
*   Cosine similarity for features: [`cosine_similarity`](utils.py)
*   Array manipulation: [`stretch_array`](utils.py), [`smoothing`](utils.py), [`norm`](utils.py) (normalization also in [`synth.py`](synth.py))
*   Plotting utilities: [`plot_with_derivatives`](utils.py), [`plot_heatmap`](utils.py)
*   Audio playback in Jupyter notebooks: `display` (using `IPython.display.Audio`)

### Audio Embeddings & Models
*   **DAC (Descript Audio Codec)**: Integration via [`DACSound`](dacsound.py) class in [`dacsound.py`](dacsound.py) for audio encoding/decoding.
*   **Jupyter Notebooks for Exploration**:
    *   [`cqt.ipynb`](cqt.ipynb): Constant-Q Transform analysis.
    *   [`mfcc.ipynb`](mfcc.ipynb): MFCC feature extraction.
    *   [`spectrum.ipynb`](spectrum.ipynb): General spectral analysis.
    *   [`music2latent.ipynb`](music2latent.ipynb): Exploring music to latent space mappings.
    *   [`dac.ipynb`](dac.ipynb): Experiments with the DAC model.
    *   [`vggish.ipynb`](vggish.ipynb): Using VGGish audio embeddings (requires `weights/audioset-vggish-3.pb`).
    *   [`playground.ipynb`](playground.ipynb): General experimentation space.
