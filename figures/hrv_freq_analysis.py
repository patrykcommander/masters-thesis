import numpy as np
import neurokit2 as nk

file = "./aidmed_ecgs/with_resp/38_ecg.npy"

ecg = np.load(file)
ecg = ecg[:,0] # load just first channel

peaks, info = nk.ecg_peaks(ecg, sampling_rate=250)
hrv_welch = nk.hrv_frequency(peaks, sampling_rate=100, show=True, psd_method="welch")
print(hrv_welch)