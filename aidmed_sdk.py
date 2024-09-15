import asyncio
import os
from datetime import datetime
from typing import List
import numpy as np
from aidlab import DeviceDelegate, Device, AidlabManager, DataType


DEST = "E:\\ml-data\\masters-thesis\\myDataset\\Patryk\\ecg_with_resp"

class MainManager(DeviceDelegate):
    def __init__(self):
        self.ecg = []
        self.resp = []

        print("Creating examination file references...")
        self.file_name_ecg = os.path.join(DEST, "6_ecg.npy")
        self.file_name_resp = os.path.join(DEST, "6_resp.npy")

    def did_connect(self, device: Device):
        print("Connected to:", device.address)

    async def run(self):
        devices = await AidlabManager().scan()
        if len(devices) > 0:
            print("Connecting to: ", devices[0].address)
            await devices[0].connect(self, [DataType.ECG, DataType.RESPIRATION, DataType.RESPIRATION_RATE])
            print("Rozpoczynam zapis danych...")
            await asyncio.sleep(150)
            print("Kończę pomiar...")
            self.save_examination()
            print("Czas pomiaru ECG", len(self.ecg) / 250)
            print("Czas pomiaru RESP", len(self.resp) / 50)
    
    def did_receive_ecg(self, device, timestamp: int, values):
        self.ecg.append([float(values[0]), timestamp])

    def did_receive_respiration(self, device: Device, timestamp: int, values: List[float]):
        self.resp.append([float(values[0]), timestamp])

    def did_disconnect(self, device: Device):
        print("Disconnect")
        self.save_examination()
    
    def save_examination(self):
        np.save(self.file_name_ecg, self.ecg)
        np.save(self.file_name_resp, self.resp)


asyncio.run(MainManager().run())
    