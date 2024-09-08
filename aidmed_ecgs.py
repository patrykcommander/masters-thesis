import os 
import numpy as np
from customLib.preprocess import *
from customLib.dataset import label_ecgs
from customLib.vis import plot_ecg
from tqdm import tqdm

x = []

smoothen = False
denoise = False
norm = False
downsample = True
total_length = 0
fs = 250
window_in_seconds = 10

x = np.empty((0, window_in_seconds * fs)) 

for root, dirs, files in os.walk("./aidmed_ecgs", topdown=False):
    for name in files:
        if root[-1].isnumeric():
            ecg = np.load(os.path.join(root, name))
            print("Processing ", name)
            total_length += ecg.shape[0]
            if smoothen:
                ecg = myConv1D(signal=ecg, kernel_length=5, padding="same")
            ecgs = split_signal(ecg, start=0, window_in_seconds=window_in_seconds, fs=fs, overlap_factor=0, normalize=norm, denoise=denoise)
            ecgs = np.array(ecgs)  # Ensure ecgs is an array
            x = np.concatenate((x, ecgs), axis=0)  # Concatenate directly

print("Total ecgs time [s]: ", x.shape[0] * x.shape[1] / fs)
y = label_ecgs(ecgs=x, sampling_rate=fs)

if downsample:
    # downsample signals
    res_x = []
    res_y = []

    for i in range(x.shape[0]):
        res_x.append(resample_signal(x[i], 1000))

    for i in range(y.shape[0]):
        res_y.append(downsample_r_peaks_probability(r_peaks_probability=y[i], original_fs=250, target_fs=100))

    res_x = np.array(res_x)
    res_y = np.array(res_y)

    np.save("./aidmed_ecgs/raw/downsampled/x.npy", res_x)
    np.save("./aidmed_ecgs/raw/downsampled/y.npy", res_y)

    print("X shape: ", res_x.shape)
    print("Y shape: ", res_y.shape)

else:
    np.save("./aidmed_ecgs/raw/x.npy", x)
    np.save("./aidmed_ecgs/raw/y.npy", y)

    print("X shape: ", x.shape)
    print("Y shape: ", y.shape)

