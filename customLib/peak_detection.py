# https://www.kaggle.com/code/stetelepta/exploring-heart-rate-variability-using-python
import numpy as np
import neurokit2 as nk
from scipy.ndimage import label

def _detect_peaks(signal: np.ndarray, threshold=0.3, qrs_filter=None):

    if qrs_filter == None:
        t = np.linspace(1.5 * np.pi, 3.5 * np.pi, 15)
        qrs_filter = np.sin(t)
    
    # standardize the ECG
    signal = (signal - signal.mean()) / signal.std()

    correlation = np.correlate(signal, qrs_filter, mode='same')
    correlation = correlation / np.max(correlation)

    return np.where(correlation > threshold)[0], correlation

def group_peaks(indices, threshold=5):  
    output = np.empty(0)

    # label groups of sample that belong to the same peak
    peak_groups, num_groups = label(np.diff(indices) < threshold)

    # iterate through groups and take the mean as peak index
    for i in np.unique(peak_groups)[1:]:
        peak_group = indices[np.where(peak_groups == i)]
        output = np.append(output, np.median(peak_group))
        
    return output.astype(int)

def detect_peaks(signal, threshold=0.3):
    r_peaks, similarity = _detect_peaks(signal, threshold=threshold)
    peaks = group_peaks(r_peaks)
    return peaks


def detect_nk(ecg_slice, fs):
    _, r_peaks = nk.ecg_peaks(ecg_slice, sampling_rate=fs)
    r_peaks = r_peaks['ECG_R_Peaks']
    return r_peaks


