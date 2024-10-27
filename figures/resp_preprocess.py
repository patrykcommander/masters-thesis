import sys

# append location of the customLib to the sys.path
customLibPath = "C:\\Users\\patry\\OneDrive\\Elearning - PG\\WETI\\masters-thesis"
sys.path.append(customLibPath)

import pickle
import numpy as np
import physio
import neurokit2 as nk
from scipy.special import log_softmax
import matplotlib.pyplot as plt
from customLib.preprocess import myConv1D, detect_local_extrema, split_signal, norm_min_max

PATH = "C:\\Users\\patry\\OneDrive\\Elearning - PG\\WETI\\masters-thesis\\aidmed_ecgs\\with_resp\\test_3"
aidmed_resp_sampling_rate=50
window_in_seconds = 60
windows_overlap_factor = 0

resp =  PATH + "\\63_resp.npy"
resp_peaks_annotation = PATH + "\\63_resp.peaks"
resp_cycle_annotation = PATH + "\\63_resp.cycles"

resp = np.load(resp)[:,0]

with open(resp_peaks_annotation, "rb") as f:
    resp_peak_indices = pickle.load(f)

with open(resp_cycle_annotation, "rb") as f:
    resp_cycle_indices = pickle.load(f)

# Create an annotation for the respiratory peaks
annotation = np.zeros((resp.shape[0]))
annotation[resp_peak_indices] = 1

resp_cycle_annotation = np.zeros((resp.shape[0]))
resp_cycle_annotation[resp_cycle_indices] = 1

# Split signals into windows
resp_windows = split_signal(resp, start=0, window_in_seconds=window_in_seconds, fs=aidmed_resp_sampling_rate, overlap_factor=windows_overlap_factor, normalize=False, denoise=False)  
peaks_annotation_windows = split_signal(annotation, start=0, window_in_seconds=window_in_seconds, fs=aidmed_resp_sampling_rate, overlap_factor=windows_overlap_factor, normalize=False, denoise=False)
cycle_annotation_windows = split_signal(resp_cycle_annotation, start=0, window_in_seconds=window_in_seconds, fs=aidmed_resp_sampling_rate, overlap_factor=windows_overlap_factor, normalize=False, denoise=False)

resp_windows = np.array(resp_windows)
peaks_annotation_windows = np.array(peaks_annotation_windows)
cycle_annotation_windows = np.array(cycle_annotation_windows)

for i in range(len(resp_windows)):
    clean_resp = nk.rsp.rsp_clean(resp_windows[i], sampling_rate=50)
    clean_resp = myConv1D(clean_resp, kernel_length=100, padding="same")
    resp_windows[i] = norm_min_max(clean_resp, 0, 1)

# KLDivLoss and jensen - shannon
# resp_windows = softmax(resp_windows, axis=-1)
resp_windows = log_softmax(resp_windows, axis=-1) # jensen-shannon, KLDivLoss normalize to (0,1) -> probability dist

t = np.array([x * 1 / aidmed_resp_sampling_rate for x in range(len(resp_windows[0]))])

fig, axs = plt.subplots(nrows=3, sharex=True)

plt.subplots_adjust(#left=0.1,
                    #bottom=0.1, 
                    #right=0.9, 
                    #top=0.9, 
                    wspace=0.2, 
                    hspace=0.3)

peaks = peaks_annotation_windows[0]
cycles = cycle_annotation_windows[0]

peaks_indices = np.where(peaks == 1)[0]

ax = axs[0]
ax.plot(t, resp[:60 * aidmed_resp_sampling_rate], color='black')
ax.set_title("Oddech", fontweight="semibold")
ax.set_ylabel("Kaniula nosowa", fontweight="semibold", labelpad=14)
ax.grid()

ax = axs[1]
ax.plot(t, resp_windows[0], 'green')
ax.set_title("Sygnał po przetworzeniu", fontweight="semibold")
ax.set_ylabel("log_softmax", fontweight="semibold")
ax.grid()

ax = axs[2]
ax.plot(t, peaks, color='dodgerblue')
ax.plot(t, cycles, color='magenta')
ax.set_title("Adnotacje", fontweight="semibold")
ax.set_ylabel("Klasa próbki", fontweight="semibold", labelpad=10)
ax.legend(["Pik sygnału", "Cykl oddechowy"])
ax.grid()

fig.supxlabel("Czas [s]", fontweight="semibold")
fig.set_size_inches(8,8)
plt.show()
