import os
import warnings
import numpy as np
from tqdm import tqdm
import neurokit2 as nk
from customLib.preprocess import norm_min_max, dwt_denoise, resample_signal

def read_dataset(path, is_validation_set=False):
  if os.path.exists(os.path.join(path, "x_train.npy")):
    x_train = np.load(os.path.join(path, "x_train.npy"))
    y_train = np.load(os.path.join(path, "y_train.npy"))
    x_test = np.load(os.path.join(path, "x_test.npy"))
    y_test = np.load(os.path.join(path, "y_test.npy"))

    if is_validation_set:
      if os.path.exists(os.path.join(path, "x_val.npy")):
        x_val = np.load(os.path.join(path, "x_val.npy"))
        y_val = np.load(os.path.join(path, "y_val.npy"))
        return (x_train, y_train, x_test, y_test, x_val, y_val)
      else:
        warnings.warn("Validation set not found. Returning only Train and Test sets.")

    return (x_train, y_train, x_test, y_test)
  else:
    print("Files not found...")

  return None

def split_dataset(x=None, y=None, split_ratio=0.8, is_validation_set=False, shuffle=False, path=None):
  if path is not None:
    if(not (os.path.isdir(path))):
      os.mkdir(path)
  else:
    warnings.warn("Path is not specified. The dataset is not being saved.")
  
  if x is None or y is None:
    raise ValueError("X or Y are empty.")

  total_ecgs = x.shape[0]  
  print(f"Total X: {x.shape[0]}")

  split_idx = int(total_ecgs * split_ratio)

  if shuffle:
    indices = np.arange(total_ecgs)
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

  x_train = np.array(x[:split_idx])
  y_train = np.array(y[:split_idx])

  x_test = np.array(x[split_idx:])
  y_test = np.array(y[split_idx:])

  if path is not None:
    print("Saving dataset to: \n", path)
    np.save(os.path.join(path, "x_train.npy"), x_train)
    np.save(os.path.join(path, "y_train.npy"), y_train)
    np.save(os.path.join(path, "x_test.npy"), x_test)
    np.save(os.path.join(path, "y_test.npy"),  y_test)

  if is_validation_set:
    total_x_test_samples = x_test.shape[0]
    val_split_idx = int(total_x_test_samples * 0.5)

    val_indices = np.arange(total_x_test_samples, dtype=int)
    np.random.shuffle(val_indices)
    val_indices = val_indices[:val_split_idx]

    x_val = x_test[val_indices]
    y_val = y_test[val_indices]

    if path is not None:
      np.save(os.path.join(path, "x_val.npy"), x_val)
      np.save(os.path.join(path, "y_val.npy"),  y_val)

    return (x_train, y_train, x_test, y_test, x_val, y_val)
  else:
    return (x_train, y_train, x_test, y_test)

# function for annotating ECGs with neurokit2
def label_ecgs(ecgs, sampling_rate=100):
  # ecgs is an array of preprocessed ECGs
  y = []

  print(f"Total ECGs: {ecgs.shape[0]}")
  
  for idx, ecg in tqdm(enumerate(ecgs), total=ecgs.shape[0]):
    try:
      _, r_peaks = nk.ecg_peaks(ecg, sampling_rate=sampling_rate)
      r_peaks_indices = r_peaks["ECG_R_Peaks"]

      r_peaks = np.zeros_like(ecg)
      r_peaks[r_peaks_indices] = 1

      y.append(r_peaks)
    except Exception as e:
      print(f"Omitting ECG number {idx + 1}")
      print(e)

  y = np.array(y)

  return y