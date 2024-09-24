import asyncio
import os
from datetime import datetime
from typing import List
import numpy as np
from aidlab import DeviceDelegate, Device, AidlabManager, DataType
import time


DEST = "E:\\ml-data\\masters-thesis\\myDataset\\Patryk\\ecg_with_resp"

class MainManager(DeviceDelegate):
    def __init__(self):
        self.ecg = []
        self.resp = []

        print("Creating examination file references...")
        self.file_name_ecg = os.path.join(DEST, "30_ecg.npy")
        self.file_name_resp = os.path.join(DEST, "30_resp.npy")

    async def run(self):
        devices = await AidlabManager().scan()
        if len(devices) > 0:
            print("Connecting to: ", devices[0].address)
            await devices[0].connect(self)
            print("Rozpoczynam zapis danych...")
            await asyncio.sleep(180)
            print("Kończę pomiar...")
            self.save_examination()
            print("Czas pomiaru ECG", len(self.ecg) / 250)
            print("Czas pomiaru RESP", len(self.resp) / 50)

    async def did_connect(self, device: Device):
        print("Connected to:", device.address)
        await device.collect([DataType.ECG, DataType.RESPIRATION], [])

    def did_disconnect(self, device: Device):
        print("Disconnected from:", device.address)
        self.save_examination()
        
    def did_receive_ecg(self, device, timestamp: int, value):
        self.ecg.append([float(value), timestamp])
        
    def did_receive_respiration(self, device: Device, timestamp: int, value: float):
        self.resp.append([float(value), timestamp])
    
    def save_examination(self):
        np.save(self.file_name_ecg, self.ecg)
        np.save(self.file_name_resp, self.resp)


asyncio.run(MainManager().run())

    