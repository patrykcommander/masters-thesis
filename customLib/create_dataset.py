import numpy as np
from tqdm import tqdm
import neurokit2 as nk
from customLib.preprocess import norm_min_max

# function for annotating ECGs with neurokit2
def label_ecgs(ecgs, sampling_rate=100):
  x = []
  y = []

  print(f"Total ECGs: {ecgs.shape[0]}")
  
  for idx, ecg in tqdm(enumerate(ecgs), total=ecgs.shape[0]):
    try:
      ecg = norm_min_max(signal=ecg, lower=-1, upper=1)
      _, r_peaks = nk.ecg_peaks(ecg, sampling_rate=sampling_rate)
      r_peaks_indices = r_peaks["ECG_R_Peaks"]

      r_peaks = np.zeros_like(ecg)
      r_peaks[r_peaks_indices] = 1

      y.append(r_peaks)
      x.append(ecg)
    except Exception as e:
      print(f"Omitting ECG number {idx + 1}")
      print(e)

  x = np.array(x)
  y = np.array(y)

  return x, y