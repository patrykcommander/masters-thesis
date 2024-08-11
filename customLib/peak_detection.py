# https://www.kaggle.com/code/stetelepta/exploring-heart-rate-variability-using-python
import math
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


import numpy as np

def find_mean_avg_r_peak_indices(y_pred):
    if len(y_pred.shape) > 1:
        y_pred = y_pred.flatten()

    result = np.zeros_like(y_pred)
    
    i = 0
    while i < len(y_pred):
        if y_pred[i] == 1:
            start = i
            while i < len(y_pred) and y_pred[i] == 1:
                i += 1
            end = i - 1
            center = math.ceil((start + end) // 2)
            result[center] = 1
        else:
            i += 1
            
    return result

def correct_prediction_according_to_aami(y_true, y_pred, sampling_rate=100):
    y_pred = find_mean_avg_r_peak_indices(y_pred)

    neighbourhood = int(0.075 * sampling_rate) # AAMI standard
    
    for i in range(len(y_true)):
        if y_true[i] == 1:
            left = max(i - neighbourhood, 0)
            right = min(i + neighbourhood + 1, len(y_true))
            subset = y_pred[left:right]
            
            if np.any(subset == 1):
                idx = np.where(subset == 1)[0][0]
                if 0 <= left + idx < len(y_pred):
                    y_pred[left + idx] = 0
                y_pred[i] = 1
                
    return y_pred