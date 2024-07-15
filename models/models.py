import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from models.datasets import ECGDataset
from sklearn.metrics import f1_score, confusion_matrix
from models.st_res_blocks import ST, ResPath, UpSampleBlock
from models.losses import WeightedBCELoss

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Training will be performed with:', self.device)
        self.to(self.device)

    def forward(self, x):
        pass

    def train_model(self, x_train, y_train, epochs=10, batch_size=1, x_val=None, y_val=None,):
        self.batch_size = batch_size
        dataset = ECGDataset(x_train, y_train)
        train_loader = DataLoader(dataset, batch_size, shuffle=False)

        if x_val is not None:
            validation_dataset = ECGDataset(x_val, y_val)
            validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(epochs):
            running_loss = 0.0
            num_r_peaks = 0.0
            num_correct = 0.0

            all_outputs = []
            all_labels = []

            self.train()
            for i, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self(x)

                loss = self.criterion(outputs, y)
                running_loss += loss.item()

                loss.backward()
                self.optimizer.step()

                outputs = outputs.cpu().detach().numpy()
                y = y.cpu().detach().numpy()

                num_r_peaks += np.where(y == 1)[0].shape[0]
                num_correct += np.where((outputs > 0.5) & (y == 1))[0].shape[0]

                all_outputs.extend(outputs.flatten())
                all_labels.extend(y.flatten())
            
            all_outputs = np.array(all_outputs)
            all_labels = np.array(all_labels)
            y_pred_binary = (all_outputs > 0.5).astype(int)

            print(f"====Epoch [{epoch + 1}/{epochs}]====")
            print(f"\nTrain Loss: {running_loss / len(train_loader):.4f}")
            self.calculate_metrics(num_correct, num_r_peaks, all_labels, y_pred_binary, phase="Train")
        
            if x_val is not None:
                self.validate(validation_loader)
  
    def validate(self, validation_loader):
        self.eval()
        running_vloss = 0.0
        num_r_peaks = 0.0
        num_correct = 0.0

        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for i, (x_val, y_val) in tqdm(enumerate(validation_loader), total=len(validation_loader)):
                x, y = x_val.to(self.device), y_val.to(self.device)
                outputs = self(x)

                loss = self.criterion(outputs, y)
                running_vloss += loss.item()

                outputs = outputs.cpu().detach().numpy()
                y = y.cpu().detach().numpy()

                num_r_peaks += np.where(y == 1)[0].shape[0]
                num_correct += np.where((outputs > 0.5) & (y == 1))[0].shape[0]

                all_outputs.extend(outputs.flatten())
                all_labels.extend(y.flatten())

            all_outputs = np.array(all_outputs)
            all_labels = np.array(all_labels)
            y_pred_binary = (all_outputs > 0.5).astype(int)

            print(f"\nValidation Loss: {running_vloss / len(validation_loader):.4f}")
            self.calculate_metrics(num_correct, num_r_peaks, all_labels, y_pred_binary, phase="Validation")
    
    def test_model(self, x_test, y_test, plot=False):
        test_dataset = ECGDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
        
        running_loss = 0.0
        num_r_peaks = 0.0
        num_correct = 0.0

        all_outputs = []
        all_labels = []
        
        self.eval()
        with torch.no_grad():
            for i, (x_test, y_test) in tqdm(enumerate(test_loader), total=len(test_loader)):
                x, y = x_test.to(self.device), y_test.to(self.device)
                outputs = self(x)

                loss = self.criterion(outputs, y)
                running_loss += loss.item()

                outputs = outputs.cpu().detach().numpy()
                y = y.cpu().detach().numpy()

                num_r_peaks += np.where(y == 1)[0].shape[0]
                num_correct += np.where((outputs > 0.5) & (y == 1))[0].shape[0]

                all_outputs.extend(outputs.flatten())
                all_labels.extend(y.flatten())

                if plot and (i % (len(test_loader) / 10) == 0):
                    ecg = x[0].cpu().detach().numpy().flatten()
                    gt = y[0].flatten()
                    pred = outputs[0].flatten()

                    plt.figure()
                    plt.plot(ecg)
                    plt.plot(gt)
                    plt.plot(pred)
                    plt.legend(["ECG", "Ground Truth", "Prediction"])
                    plt.grid()
                    plt.show()
                
        all_outputs = np.array(all_outputs)
        all_labels = np.array(all_labels)
        y_pred_binary = (all_outputs > 0.5).astype(int)

        print(f"\nTest Loss: {running_loss / len(test_loader):.4f}")
        self.calculate_metrics(num_correct, num_r_peaks, all_labels, y_pred_binary, phase="Test")
    
    # we only care about the precision of the R_peaks (binary class 1) and we about the false positive rate
    def calculate_metrics(self, num_correct_peaks, total_peaks, y_true, y_pred_binary, phase="Train"):
        accuracy = num_correct_peaks / total_peaks * 100

        f1 = f1_score(y_true, y_pred_binary)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        print(f"{phase} Accuracy: {accuracy:.5f} %")
        print(f"{phase} F1 Score: {f1:.5f}")
        print(f"{phase} TPR: {tpr:.5f}")
        print(f"{phase} FPR: {fpr:.5f}\n")

class ST_RES_NET(BasicModel):
    def __init__(self, learning_rate=1e-4):
        super(ST_RES_NET, self).__init__()
        self.st_block_1 = ST(1, 8)
        self.st_block_2 = ST(8, 16)
        self.st_block_3 = ST(16, 32)
        self.st_block_4 = ST(32, 64)

        self.res_path_1 = ResPath(8, 8, num_parallel_convs=4)
        self.res_path_2 = ResPath(16, 16, num_parallel_convs=3)
        self.res_path_3 = ResPath(32, 32, num_parallel_convs=2)
        self.bottleneck = ResPath(64, 64, num_parallel_convs=1)

        self.upsample_1 = UpSampleBlock(64, 32)
        self.upsample_2 = UpSampleBlock(32, 16)
        self.upsample_3 = UpSampleBlock(16, 8)

        self.output = nn.Conv1d(8, 1, kernel_size=1, stride=1)

        self.criterion = WeightedBCELoss() #nn.BCELoss()
        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        self.to(self.device)

    def forward(self, x):
        f1, p1 = self.st_block_1(x)
        f2, p2 = self.st_block_2(p1)
        f3, p3 = self.st_block_3(p2)
        f4, _ = self.st_block_4(p3)

        f1 = self.res_path_1(f1)
        f2 = self.res_path_2(f2)
        f3 = self.res_path_3(f3)

        embedding = self.bottleneck(f4) # 64

        u1, _ = self.upsample_1(embedding, f3) # 32
        u2, _ = self.upsample_2(u1, f2) # 16
        u3, _ = self.upsample_3(u2, f1) # 8

        output = self.output(u3)

        return torch.sigmoid(output)

class LSTM(BasicModel):
    def __init__(self, input_size, hidden_size, learning_rate=1e-2):
      super(LSTM, self).__init__()
      self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, dropout=0.3)
      self.relu = torch.nn.ReLU()
      # achieveing better results with Dense layer instead of Conv1d and when dropout is used after lstm instead of Dense layer
      #self.conv = torch.nn.Conv1d(kernel_size=1, in_channels=hidden_size, out_channels=1) 
      self.flatten = torch.nn.Flatten()
      self.dense = torch.nn.Linear(in_features=hidden_size, out_features=1)
      self.sigmoid = torch.nn.Sigmoid()

      self.criterion = torch.nn.BCELoss() # WeightedBCELoss()
      self.optimizer = Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
      x, _ = self.lstm(x)
      x = self.relu(x)
      x = self.dense(x)
      output = self.sigmoid(x)
      return output


