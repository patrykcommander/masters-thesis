import asyncio
import os
from datetime import datetime
import numpy as np
from aidlab import AidlabManager, Device, DeviceDelegate, DataType


DEST = "E:\\ml-data\\masters-thesis\\myDataset\\Patryk"

class MainManager(DeviceDelegate):
    def did_connect(self, device: Device):
        print("Connected to:", device.address)
        print("Creating examination file....")
        self.current_ecg = []
        self.current_rr = []
        current_date = datetime.now().strftime("%d-%m-%y_%H_%M")
        self.file_name_ecg = os.path.join(DEST, (current_date + ".npy"))
        self.file_name_rr = os.path.join(DEST, (current_date + "rr.npy"))
        #if not os.path.isfile(self.file_name):
        #    self.fp = open(self.file_name, '+a')

    async def run(self):
        devices = await AidlabManager().scan()
        if len(devices) > 0:
            print("Connecting to: ", devices[0].address)
            await devices[0].connect(self, [DataType.ECG, DataType.RR])
            print("Rozpoczynam zapis danych EKG...")
            await asyncio.sleep(600)
            print("Kończę pomiar...")
            self.save_examination()
            print(len(self.current_ecg))
            print(len(self.current_rr))
    
    def did_receive_ecg(self, _, timestamp, values):
        self.current_ecg.append(float(values[0]))

    def did_receive_rr(self, _, timestamp, values):
        self.current_rr.append(float(values))

    def did_disconnect(self, device: Device):
        print("Disconnect")
        self.save_examination()
    
    def save_examination(self):
        np.save(self.file_name_ecg, self.current_ecg)
        np.save(self.file_name_rr, self.current_rr)


asyncio.run(MainManager().run())
    