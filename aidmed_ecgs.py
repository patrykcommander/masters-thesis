import os 
import numpy as np
from customLib.preprocess import *
from customLib.dataset import label_ecgs
from customLib.vis import plot_ecg
from tqdm import tqdm

x = []

smoothen = True
denoise = False
norm = True
total_length = 0
x = np.empty((0, 2500)) 

for root, dirs, files in os.walk("./aidmed_ecgs", topdown=False):
    for name in files:
        if not root.endswith("preprocessed"):
            ecg = np.load(os.path.join(root, name))
            print("Processing ", name)
            total_length += ecg.shape[0]
            ecg = myConv1D(signal=ecg, kernel_length=5, padding="same")
            ecgs = split_signal(ecg, start=0, window_in_seconds=10, fs=250, overlap_factor=0, normalize=True, denoise=True)
            ecgs_array = np.array(ecgs)  # Ensure ecgs is an array
            x = np.concatenate((x, ecgs_array), axis=0)  # Concatenate directly

print("Total ecgs time [s]: ", total_length / 250)
y = label_ecgs(ecgs=x, sampling_rate=250)

np.save("./aidmed_ecgs/preprocessed/x.npy", x)
np.save("./aidmed_ecgs/preprocessed/y.npy", y)
