import sys

# append location of the customLib to the sys.path
customLibPath = ""
sys.path.append(customLibPath)

import numpy as np
import physio
import neurokit2 as nk
import matplotlib.pyplot as plt
from customLib.preprocess import myConv1D, detect_local_extrema



PATH = "C:\\Users\\patry\\OneDrive\\Elearning - PG\\WETI\\masters-thesis\\aidmed_ecgs\\with_resp"
aidmed_resp_sampling_rate=50


resp_file = PATH + "\\50_resp.npy"
resp = np.load(resp_file)
resp = resp[:,0]
resp = myConv1D(resp, 100, "same")


x = [x for x in range(len(resp))]

local_maxima, local_minima = detect_local_extrema(x, resp)
local_maxima = local_maxima[::2]
local_minima = local_minima[::2]

_, cycles = physio.compute_respiration(resp, aidmed_resp_sampling_rate, parameter_preset='human_airflow')
inspi_index = cycles['inspi_index'].values
expi_index = cycles['expi_index'].values
cycles = np.sort(np.concatenate((inspi_index, expi_index)))
split = mean_index = ((inspi_index + expi_index) / 2).astype(int)

peaks = nk.rsp_findpeaks(resp, sampling_rate=50)['RSP_Peaks']

t = np.array([x * 1/ aidmed_resp_sampling_rate for x in range(len(resp))])

fig, axs = plt.subplots(nrows=2, sharex=True)

plt.subplots_adjust(#left=0.1,
                    #bottom=0.1, 
                    #right=0.9, 
                    #top=0.9, 
                    wspace=0.2, 
                    hspace=0.2)

ax = axs[0]
ax.plot(t, resp, color='black')
ax.scatter(t[peaks], resp[peaks], s=50, color="red", marker="o")
ax.scatter(t[split], resp[split], s=50, color="green", marker="o")
ax.scatter(t[inspi_index], resp[inspi_index], s=50, color="dodgerblue", marker="s")
ax.scatter(t[expi_index], resp[expi_index], s=50, color="purple", marker="s")
ax.grid()
ax.legend(["Oddech", "Maksimum", "Separacja cyklu", "Wydech", "Wdech"])

ax = axs[1]
ax.plot(t, resp, 'black')
ax.scatter(t[local_maxima], resp[local_maxima], s=50, color="lime", marker="o")
ax.scatter(t[local_minima], resp[local_minima], s=50, color="violet", marker="o")
ax.grid()
ax.legend(["Oddech", "Lokalne maksimum", "Lokalne minimum"])
ax.set_title("")

fig.suptitle("Przykładowe adnotacje oddechu", fontweight="semibold", y=0.92)
fig.supxlabel("Czas [s]", fontweight="semibold", y=0.05)
fig.supylabel("Sygnał z kaniuli nosowej", fontweight="semibold", x=0.05)
fig.set_size_inches(8,8)
plt.show()
