import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
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

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def plot_metrics(metrics):
    fig, axs = plt.subplots(nrows=1, ncols=3)
    plt.subplots_adjust(
        wspace=0.3
    )


    epochs = [x+1 for x in range(len(metrics["train"]["loss"]))]
    ticks = [x for x in epochs if x % 5 == 0]
    ticks.insert(0, 1)

    loss_formatter = FormatStrFormatter('%.2f')  # Rounds loss to 2 decimal places
    other_formatter = FormatStrFormatter('%.3f')  # Rounds other metrics to 3 decimal places

    # Loss
    ax = axs[0]
    ax.plot(epochs, metrics["train"]["loss"], color='black', linestyle="solid")
    ax.plot(epochs, metrics["validation"]["loss"], color='red', linestyle='dashed')
    ax.grid()
    ax.set_title("Funkcja straty", fontweight="semibold")
    ax.set_xticks(ticks)
    ax.yaxis.set_major_formatter(loss_formatter)  # Use 2 decimal rounding for loss

    # F1
    ax = axs[1]
    ax.plot(epochs, metrics["train"]["f1"], color='black', linestyle="solid")
    ax.plot(epochs, metrics["validation"]["f1"], color='red', linestyle='dashed')
    ax.grid()
    ax.set_title("Metryka F1", fontweight="semibold")
    ax.set_xticks(ticks)
    ax.yaxis.set_major_formatter(other_formatter)  # Use 3 decimal rounding for f1

    # Accuracy
    ax = axs[2]
    ax.plot(epochs, metrics["train"]["accuracy"], color='black', linestyle="solid")
    ax.plot(epochs, metrics["validation"]["accuracy"], color='red', linestyle='dashed')
    ax.grid()
    ax.set_title("Dokładność", fontweight="semibold")
    ax.set_xticks(ticks)
    ax.yaxis.set_major_formatter(other_formatter)  # Use 3 decimal rounding for accuracy

    fig.legend(["Trening", "Walidacja"], loc="lower left", ncol=2, bbox_to_anchor=(0.4, -0.12))
    fig.supxlabel("Epoka", fontweight="semibold", y =-0.02)
    fig.supylabel("Wartości", fontweight="semibold", x=0.06)
    fig.set_size_inches(w=12, h=4)

