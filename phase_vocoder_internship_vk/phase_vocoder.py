import os
import sys

import librosa
import numpy as np
import soundfile


N_FFT = 2048
HOP_LENGTH = N_FFT // 4


def phase_vocoder(wav: np.array,
                  time_stratch_ratio: float,
                  n_fft=N_FFT,
                  hop_length=HOP_LENGTH) -> np.array:
    """
    phase vocoder algorithm implemented according to the article:
    http://www.guitarpitchshifter.com/algorithm.html
    """
    D = librosa.stft(wav,
                     n_fft=n_fft,
                     hop_length=hop_length,
                     window='hann')

    omega_bin = np.linspace(0, np.pi * hop_length, D.shape[0])
    phi_s = np.angle(D[:, 0])
    
    time_steps = np.arange(0, D.shape[1], time_stratch_ratio, dtype=np.float64)
    D_new = np.empty((D.shape[0], len(time_steps)), D.dtype)
    
    D = np.pad(D, [(0, 0), (0, 2)], mode='constant')
    
    for i, step in enumerate(time_steps):
        columns = D[:, int(step):int(step)+2]
        
        a = np.mod(step, 1)
        magnitude = (1 - a) * np.abs(columns[:, 0]) + a * np.abs(columns[:, 1])
        D_new[:, i] = magnitude * np.exp(1.0j * phi_s)
        
        d_omega = np.angle(columns[:, 1]) - np.angle(columns[:, 0]) - omega_bin
        d_omega_wrapped = np.mod((d_omega + np.pi), 2 * np.pi) - np.pi
        omega_true = omega_bin + d_omega_wrapped
        phi_s += omega_true

    wav_new = librosa.istft(D_new,
                            hop_length=hop_length,
                            window='hann')

    return wav_new


def main(argv):
    input_path = argv[1]
    output_path = argv[2]
    time_stratch_ratio = float(argv[3])
    
    wav, sr = librosa.load(input_path)
    wav_new = phase_vocoder(wav, time_stratch_ratio)
    soundfile.write(output_path, wav_new, sr)
    
    return 0


if __name__ == '__main__':
    main(sys.argv)
