import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt


def norm_min_max(signal, lower, upper):
    signal_std = (signal - signal.min(axis=0)) / (signal.max(axis=0) - signal.min(axis=0))
    signal_scaled = signal_std * (upper - lower) + lower
    return signal_scaled
        
sample = 56
aidmed_ecg_sampling_rate = 250
aidmed_resp_sampling_rate = 50

start_s = 30
stop_s = 90

start = start_s * aidmed_ecg_sampling_rate
stop = stop_s * aidmed_ecg_sampling_rate
ecg = np.load(f"./aidmed_ecgs/with_resp/{sample}_ecg.npy")
ecg = ecg[start:stop,0]
ecg = ecg.flatten()
ecg = norm_min_max(ecg, -1, 1)

_, r_peaks = nk.ecg_peaks(ecg, sampling_rate=aidmed_ecg_sampling_rate)
r_peaks = np.array(r_peaks['ECG_R_Peaks'])
rris = np.diff(r_peaks) * 1 / aidmed_ecg_sampling_rate * 1000


start = start_s * aidmed_resp_sampling_rate
stop = stop_s  * aidmed_resp_sampling_rate
resp = np.load(f"./aidmed_ecgs/with_resp/{sample}_resp.npy")
resp = resp[start:stop,0]
resp = resp.flatten()


t = np.array([(x * 1/ aidmed_ecg_sampling_rate) + start_s for x in range(len(ecg))])
t_resp = np.array([(x * 1/ aidmed_resp_sampling_rate) + start_s for x in range(len(resp))])


fig, axs = plt.subplots(nrows=4, sharex=True)

ax = axs[0]
ax.grid()
ax.plot(t, ecg, 'k-')
ax.scatter(t[r_peaks], ecg[r_peaks], color="red", s=50, marker="x")
# ax.plot(r_peaks_nk * 1/ aidmed_ecg_sampling_rate, x[sample][r_peaks_nk], 'rx')
# ax.plot(r_peaks_lstm * 1/ aidmed_ecg_sampling_rate, x[sample][r_peaks_lstm], 'gx')
ax.legend(["ECG", "Załamki R"], loc='upper right', framealpha=1)
ax.set_ylabel("Znormalizowane\nECG", fontweight="semibold", fontsize=12, labelpad=10)

ax = axs[1]
ax.plot(t_resp, resp, color='dodgerblue')
ax.grid()
ax.legend(["Oddech"], loc='upper right')
ax.set_ylabel("Kaniula\nnosowa", fontweight="semibold", fontsize=12, labelpad=18)

ax = axs[2]
ax.plot(t[r_peaks[:-1]], rris, color="forestgreen")
ax.grid()
ax.legend(["Interwały RR"], loc='upper right')
ax.set_ylabel("Odstęp czasu\n[ms]", fontweight="semibold", fontsize=12)

ax = axs[3]
ax.plot(t[r_peaks], ecg[r_peaks], color='red')
ax.grid()
ax.legend(["RPA"], loc='upper right')
ax.set_ylabel("Amplituda\nzałamka R", fontweight="semibold", fontsize=12)


fig.supxlabel("Czas [s]", fontweight="semibold", y=0.05)
fig.suptitle("Zjawisko RSA", fontweight="semibold", y=0.92)
fig.set_size_inches(10,8)
plt.show()