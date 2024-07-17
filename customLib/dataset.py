import os
import warnings
from tqdm import tqdm
import numpy as np

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
      dataset_path = os.path.join(path, "dataset")
      if(not (os.path.isdir(dataset_path))):
        os.mkdir(dataset_path)
  else:
    warnings.warn("Path is not specified. The dataset is not being saved.")
  
  if x is None or y is None:
    raise ValueError("X or Y are empty.")

  total_ecgs = x.shape[0]  
  print(f"Total ECGs: {x.shape[0]}")

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
    np.save(os.path.join(dataset_path, "x_train.npy"), x_train)
    np.save(os.path.join(dataset_path, "y_train.npy"), y_train)
    np.save(os.path.join(dataset_path, "x_test.npy"), x_test)
    np.save(os.path.join(dataset_path, "y_test.npy"),  y_test)

  if is_validation_set:
    total_x_test_samples = x_test.shape[0]
    val_split_idx = int(total_x_test_samples * 0.5)

    val_indices = np.arange(total_x_test_samples, dtype=int)
    np.random.shuffle(val_indices)
    val_indices = val_indices[:val_split_idx]

    x_val = x_test[val_indices]
    y_val = y_test[val_indices]

    if path is not None:
      np.save(os.path.join(dataset_path, "x_val.npy"), x_val)
      np.save(os.path.join(dataset_path, "y_val.npy"),  y_val)

    return (x_train, y_train, x_test, y_test, x_val, y_val)
  else:
    return (x_train, y_train, x_test, y_test)
