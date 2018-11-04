# Highly reference from
# `https://github.com/Kyubyong/tacotron/blob/master/utils.py`
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy
import unicodedata
import re
import numpy as np
import scipy
import librosa
from hyperparams import Hyperparams as hps


#### Audio ####
def load_audio(audio_file, sr):
    y, sr = librosa.load(audio_file, sr)
    return y, sr


def get_spectrogram(audio):
    """Return log(mel-spectrogram) and log(magnitude) of `audio`
    Args:
        audio_file: A string. The filepath of the audio.

    Returns:
        mel: Mel-scale spectrogram. A 2d array of shape (time, n_mels)
        mag: Magnitude spectrogram. A 2d array of shape (time, 1 + n_fft/2)
    """
    y,  _ = librosa.effects.trim(audio)
    y = _pre_emphasis(y, hps.pre_emphasis)
    # Linear spectrogram
    linear = librosa.stft(y=y, n_fft=hps.n_fft,
                          hop_length=hps.hop_length, win_length=hps.win_length)
    mag = np.abs(linear)  # shape: (1 + n_fft/2, time)
    mel = _mag2mel(mag, hps.sampling_rate, hps.n_fft, hps.n_mels)
    mag = _amp_to_db(mag)
    mel = _amp_to_db(mel)
    mag = _normalize(mag, hps.ref_db, hps.max_db)
    mel = _normalize(mel, hps.ref_db, hps.max_db)
    mag = mag.T.astype(np.float32)  # shape: (time, 1 + nfft/2)
    mel = mel.T.astype(np.float32)  # shape: (time, n_mels)

    return mel, mag


def mag2wav(mag):
    """Return recovered wave file base on the given magnitude spectrogram `mag`
    """
    mag = mag.T
    mag = _de_normalize(mag, hps.ref_db, hps.max_db)
    mag = _db_to_amp(mag)
    wav = _griffin_lim(mag, hps.n_fft, hps.hop_length, hps.win_length)
    wav = _de_emphasis(wav, hps.pre_emphasis)
    wav, _ = librosa.effects.trim(wav)
    wav = wav.astype(np.float32)
    return wav

def save_wav(wav, path):
    librosa.output.write_wav(path, wav, hps.sampling_rate)

def _mag2mel(mag, sr, n_fft, n_mels):
    # shape: (n_mels, 1 + n_fft/2)
    mel_matrix = librosa.filters.mel(sr, n_fft, n_mels)
    mel = np.dot(mel_matrix, mag)  # shape: (n_mels, time)
    return mel


def _amp_to_db(signal, maximum=1e-5):
    return 20 * np.log10(np.maximum(maximum, signal))


def _db_to_amp(signal_db):
    return np.power(10.0, signal_db / 20.0)


def _normalize(signal_db, ref_db, max_db):
    return np.clip((signal_db - ref_db + max_db) / max_db, a_min=1e-8, a_max=1)


def _de_normalize(signal, ref_db, max_db):
    return (np.clip(signal, a_min=0, a_max=1) * max_db) - max_db + ref_db


def _pre_emphasis(signal, coeff):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def _de_emphasis(signal, coeff):
    return scipy.signal.lfilter([1], [1, -coeff], signal)


def _griffin_lim(mag, n_fft, hop_length, win_length):
    """Griffin-Lim algorithm
    """
    X_best = deepcopy(mag)
    for _ in range(hps.GL_n_iter):
        X_t = librosa.istft(X_best, hop_length=hop_length,
                            win_length=win_length, window='hann')
        est = librosa.stft(X_t, n_fft, hop_length, win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = mag * phase
    X_t = librosa.istft(X_best, hop_length=hop_length,
                        win_length=win_length, window='hann')
    y = np.real(X_t)
    return y


#### Text ####
def text_normalize(text):
    text = ''.join(char for char in unicodedata.normalize('NFD', text) if unicodedata.category(char) != 'Mn')  # Strip accents
    text = text.lower()
    text = re.sub("[^{}]".format(hps.vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text


def char2idx(char):
    return hps.vocab.find(char)


def sent2idx(sent):
    return [char2idx(ch) for ch in sent]

#### Plot ####
def save_spectrogram(spectrogram, path, dpi=500, format='png'):
    """Save figure of `spectrogram` to `path'
    Args:
        spectrogram: A numpy 2d array of shape (time, freq).
        path: the path to save.
    """
    plt.gcf().clear()  # Clear current previous figure
    cmap = plt.get_cmap('jet')
    t = hps.frame_len + np.arange(spectrogram.shape[0]) * hps.frame_hop
    f = np.arange(spectrogram.shape[1]) * hps.sampling_rate / hps.n_fft
    plt.pcolormesh(t, f, spectrogram.T, cmap=cmap)
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.savefig(path, dpi=dpi, format=format)

def save_alignment(alignment, step, path, dpi=500, format='png'):
    """Save figure of `alignment` ot `path`
    Args:
        alignment: A numpy 2d array of shape (decoder_time_step, encoder_seq_length).
        path: the path to save.
    """
    plt.gcf().clear()
    plt.pcolormesh(alignment.T)
    plt.colorbar()
    plt.title('Time step {}'.format(step))
    plt.xlabel('Decoder time step (frame)')
    plt.ylabel('Encoder time step (character)')
    plt.savefig(path, dpi=dpi, format=format)

