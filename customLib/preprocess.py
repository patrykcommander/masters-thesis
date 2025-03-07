import numpy as np
from scipy.signal import resample
import matplotlib.pyplot as plt
import pywt

def add_padding(signal: np.ndarray, kernel_length: int):

    n_padding = int(kernel_length/2)
    new_signal = np.zeros((signal.shape[0] + 2*n_padding))
    new_signal[n_padding:-n_padding] = signal

    # fill in the padding with boundary values
    new_signal[:n_padding] = signal[0]
    new_signal[-n_padding:] = signal[-1]

    return new_signal   

def myConv1D(signal: np.ndarray, kernel_length: int, padding='same'):
    weights = np.ones(kernel_length)
    weights /= kernel_length                # weights for smoothening the signal
    output = []

    if padding == 'same':                   # add padding
        signal = add_padding(signal, kernel_length)

    n_samples = signal.shape[0]
    index = int(kernel_length/2)

    for i in range(index, n_samples-index):
        start = i-index
        stop = start+kernel_length
        part = signal[start : stop]
        step = np.matmul(part, weights)
        output.append(step)  
    
    return np.array(output)

def norm_min_max(signal, lower, upper):
    with np.errstate(invalid='raise'):
        try:
            if signal.ndim > 2 or (signal.shape[0] > 1 and signal.ndim == 2):
                raise Exception("Invalid shape: ", signal.shape)

            signal = signal.flatten()
            signal_std = (signal - signal.min(axis=0)) / (signal.max(axis=0) - signal.min(axis=0))
            signal_scaled = signal_std * (upper - lower) + lower
            return signal_scaled
        except Exception as e:
            print(e)
            plt.plot(signal.flatten())
            plt.show()
            return 1

def stationary(signal):
    return np.diff(signal)

def split_signal(signal, normalize: tuple, start=0, window_in_seconds=10, fs=250, overlap_factor=0.1, denoise=False):
    window_size = int(fs * window_in_seconds)
    overlap = int(window_size * overlap_factor)
    step = window_size - overlap
    signal_windows = []
    
    while start + window_size <= len(signal):
        signal_slice = signal[start:start+window_size]
        start += step
        if denoise:
            signal_slice = dwt_denoise(signal_slice)
        if isinstance(normalize, (tuple)):
            if normalize[0] == True:
                signal_slice = norm_min_max(signal_slice, lower=normalize[1][0], upper=normalize[1][1])
        if np.isnan(signal_slice).any():
            try:
                raise ValueError(f"NaN values found in signal slice starting at index {start - step}")
            except ValueError as e:
                print(e)
                signal_slice = 1

        signal_windows.append(signal_slice)
    return signal_windows

def _calc_hrv(peaks_time):
    # when no R-peak is detected 
    with np.errstate(invalid='raise'):
        try:
            RRI = np.diff(peaks_time)
            SDNN = np.std(RRI).round(4)
        except:
            RRI = 0
            SDNN = 0

        return RRI, SDNN

""" def calculate_hrv(signal: np.array, fs: int, threshold=0.3):
    peaks_indices = detect_my_peaks(signal=signal, threshold=threshold)
    peaks_time = np.array([x * 1000/fs for x in peaks_indices])
    RRI, SDNN = _calc_hrv(peaks_time)
    return peaks_indices, RRI, SDNN """

def resample_signal(signal, num_samples):
    return resample(x=signal, num=num_samples)

def downsample_r_peaks_probability(r_peaks_probability, original_fs, target_fs):
    downsample_rate = target_fs / original_fs

    original_peaks = np.where(r_peaks_probability == 1)[0]
    scaled_peaks = np.round(original_peaks * downsample_rate).astype(int)
    scaled_peaks = np.unique(scaled_peaks)

    new_length = int(np.round(r_peaks_probability.shape[0] * downsample_rate))
    scaled_peaks = scaled_peaks[scaled_peaks < new_length]

    resampled_r_peak_probability = np.zeros(new_length, dtype=int)
    resampled_r_peak_probability[scaled_peaks] = 1

    return resampled_r_peak_probability


### MIT-BIH dataset
def dwt_denoise(signal, wavelet="db8"):
  coefs = pywt.wavedec(signal, wavelet)
  for i, _ in enumerate(coefs):
    if i not in [0, 1, 7, 8]:
      continue
    else:
      coefs[i] *= 0
  
  signal_denoised = pywt.waverec(coefs, wavelet)

  return signal_denoised

def expand_labels(annotation_windows: list, fileName="", left_shift=5, right_shift=5):
    # annotation_windows - list of numpy 1D arrays

    #### Expanding labels as in the paper DOI: 10.1109/TIM.2023.3241997
    #### Adding additional probabilities for each R-peak (14 samples backwards and 15 samples forwards)

    expanded_annotations = []

    sig_len = annotation_windows[0].shape[0]

    for i, window in enumerate(annotation_windows):
        new_r_peak_indices = []

        r_peaks_indices = np.where(window == 1)[0]

        for annotation_idx in r_peaks_indices:
            if annotation_idx - left_shift > 5 and annotation_idx + right_shift < sig_len - 5:
                new_r_peak_indices.extend(range(int(annotation_idx - left_shift), int(annotation_idx + right_shift)))
            elif annotation_idx - left_shift <= 5 and annotation_idx - left_shift > 0:
                new_r_peak_indices.extend(range(int(annotation_idx - (int(left_shift) / 2)), int(annotation_idx + int(right_shift) / 2)))
                print(f"Annotation index at the beginning of the ECG window from file {fileName} window {i}. Expanding the label by a smaller amount.")
            elif annotation_idx + right_shift >= sig_len - 5 and annotation_idx + right_shift < sig_len:
                new_r_peak_indices.extend(range(int(annotation_idx - (int(left_shift) / 2)), int(annotation_idx + (int(right_shift) / 2))))
                print(f"Annotation index at the end of the ECG window from file {fileName} window {i}. Expanding the label by a smaller amount.")
            else:
                new_r_peak_indices.append(annotation_idx)
                print(f"Annotation index at the very beginning or very end of the ECG window from file {fileName} window {i}. Therefore not expanding this index.")
        
        new_annotation = np.zeros(shape=(sig_len, ))
        new_annotation[new_r_peak_indices] = 1

        expanded_annotations.append(new_annotation)

    return np.array(expanded_annotations)


def invert_log_softmax(log_softmax_output: np.array) -> np.array:
    """
    Inverts the log_softmax output to recover the softmax probabilities.
    
    Args:
    - log_softmax_output: np.array, the output of log_softmax.
    
    Returns:
    - softmax_probs: np.array, the recovered softmax probabilities.
    """
    # Step 1: Exponentiate to invert the log operation
    softmax_probs = np.exp(log_softmax_output)
    
    # Step 2: Normalize to ensure probabilities sum to 1
    # This step is technically optional since the exponentiation already handles it,
    # but it's good practice to normalize to avoid any floating-point issues.
    softmax_probs /= np.sum(softmax_probs, axis=-1, keepdims=True)
    
    return softmax_probs


def detect_local_extrema(x: np.array, y: np.array):
    """
    Detects local maxima and minima using the second derivative test.
    
    Args:
    - x: np.array, independent variable (e.g., time or position)
    - y: np.array, dependent variable (e.g., function values)
    
    Returns:
    - local_maxima: list of tuples (x, y) of local maxima points
    - local_minima: list of tuples (x, y) of local minima points
    """
    # Step 1: Calculate the first and second derivatives using numpy's gradient function
    dy_dx = np.gradient(y, x)   # First derivative
    d2y_dx2 = np.gradient(dy_dx, x)  # Second derivative

    # Step 2: Identify points where the first derivative crosses zero (potential extrema)
    local_maxima = []
    local_minima = []
    
    for i in range(1, len(dy_dx) - 1):
        if np.sign(dy_dx[i-1]) != np.sign(dy_dx[i+1]):  # Derivative changes sign
            if d2y_dx2[i] < 0:  # Negative second derivative => local maxima
                local_maxima.append((x[i]))
            elif d2y_dx2[i] > 0:  # Positive second derivative => local minima
                local_minima.append((x[i]))
    
    return np.array(local_maxima), np.array(local_minima)
