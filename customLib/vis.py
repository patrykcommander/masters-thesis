import numpy as np
import matplotlib.pyplot as plt
import pywt

#plotting the signal
def plot_ecg(signal: np.ndarray, r_peaks=None, fs: int = 100, title = ""):
    t = [1/fs * x for x in range(len(signal))]
    plt.plot(figsize=(5,4))
    plt.plot(t, signal)
    if r_peaks is not None:
        if r_peaks.shape[0] == signal.shape[0]:   # r_peaks passed as probability vector (len, ) / x is of shape (1, len)
            plt.plot(t, r_peaks, "r-")
            plt.legend(["ECG", "R peaks"], loc="lower right")
        elif r_peaks.shape[0] > 0:                # r_peaks passed as indices
            r_peaks_time = r_peaks / fs
            plt.plot(r_peaks_time, signal[r_peaks], "rx")
            plt.legend(["ECG", "R peaks"], loc="lower right")
    else:
        plt.legend(["ECG"], loc="lower right")
    plt.grid(color="#858281", linestyle='--')
    plt.xlabel("Time [s]")
    if len(title) > 0:
        plt.title(title)
    plt.show()

def plot_dwt(ECG):
    coefs = pywt.wavedec(ECG, "db8")
    plt.figure(figsize=(10,10))
    for i, sig in enumerate(coefs):
        plt.subplot(5, 2, i+1)
        plt.plot(sig)
        plt.grid()
        plt.legend(["Decomposition component " + str(i+1)])
        if i == len(coefs) - 1:
            plt.subplot(5,2,10)
            plt.plot(ECG)
            plt.grid()
            plt.legend(["ECG"])
    plt.show()
