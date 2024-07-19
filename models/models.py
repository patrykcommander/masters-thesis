import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from models.datasets import ECGDataset
from sklearn.metrics import f1_score, confusion_matrix, balanced_accuracy_score
from models.st_res_blocks import ST, ResPath, UpSampleBlock
from models.losses import WeightedBCELoss

class BasicModel(nn.Module):
    def __init__(self, apply_sigmoid=False):
        super(BasicModel, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Training will be performed with:', self.device)
        self.to(self.device)

        self.apply_sigmoid = apply_sigmoid

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
            print(f"====Epoch [{epoch + 1}/{epochs}]====")
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

                if self.apply_sigmoid:
                        outputs = torch.sigmoid(outputs)

                outputs = outputs.cpu().detach().numpy()
                y = y.cpu().detach().numpy()

                num_r_peaks += np.where(y == 1)[0].shape[0]
                num_correct += np.where((outputs > 0.5) & (y == 1))[0].shape[0]

                all_outputs.extend(outputs.flatten())
                all_labels.extend(y.flatten())

                if i == len(train_loader) - 1:
                    randIdx = np.random.randint(low=0, high=x.shape[0])
                    plt.figure()
                    plt.title(f"Last batch, sample number {randIdx + 1}")
                    plt.plot(x[randIdx].cpu().detach().numpy().flatten(), 'b-')
                    plt.plot(y[randIdx].flatten(), 'g-')
                    plt.plot(outputs[randIdx].flatten(), 'r--')
                    plt.legend(["ECG", "Ground Truth", "Prediction"])
                    plt.show()
            
            all_outputs = np.array(all_outputs)
            all_labels = np.array(all_labels)
            y_pred_binary = (all_outputs > 0.5).astype(int)

            print(f"\nTrain Loss: {running_loss / len(train_loader):.4f}")
            self.calculate_metrics(all_labels, y_pred_binary, phase="Train")
        
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

                if self.apply_sigmoid:
                    outputs = torch.sigmoid(outputs)

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
            self.calculate_metrics(all_labels, y_pred_binary, phase="Validation")
    
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

                if self.apply_sigmoid:
                    outputs = torch.sigmoid(outputs)

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
        self.calculate_metrics(all_labels, y_pred_binary, phase="Test")
    
    # we only care about the precision of the R_peaks (binary class 1) and we about the false positive rate
    def calculate_metrics(self, y_true, y_pred_binary, phase="Train"):
        # TODO: upgrade metrics (R-wave prediction in the particular neighbourhood of the labeled sample treated as correct)

        total_targets = y_true.shape[0]
        positive_count = np.sum(y_true)
        negative_count = total_targets - positive_count
        w_p = negative_count / total_targets
        w_n = positive_count / total_targets

        weights = [ w_p if x == 1 else w_n for x in y_true ]

        accuracy = balanced_accuracy_score(y_true=y_true, y_pred=y_pred_binary, sample_weight=weights)
        
        f1 = f1_score(y_true, y_pred_binary)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        print(f"{phase} Weighted Accuracy: {accuracy:.5f}")
        print(f"{phase} F1 Score: {f1:.5f}")
        print(f"{phase} TPR: {tpr:.5f}")
        print(f"{phase} FPR: {fpr:.5f}")
        print(f"{phase} TNR: {tnr:.5f}")
        print(f"{phase} FNR: {fnr:.5f}\n")

class ST_RES_NET(BasicModel):
    def __init__(self, learning_rate=1e-4):
        super(ST_RES_NET, self).__init__(apply_sigmoid=False)
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

# trained with expanded_labels
class LSTM(BasicModel):
    def __init__(self, input_dim, hidden_size, lr=1e-2, loss_pos_weight=None):
      super(LSTM, self).__init__(apply_sigmoid=True)
      self.lstm_1 = torch.nn.LSTM(input_size=input_dim, hidden_size=hidden_size, batch_first=True, bidirectional=True, dropout=0.3)
      self.lstm_2 = torch.nn.LSTM(input_size=2*hidden_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)
      #self.relu = torch.nn.ReLU()
      # achieveing better results with Dense layer instead of Conv1d and when dropout is used after lstm instead of Dense layer
      #self.conv = torch.nn.Conv1d(kernel_size=1, in_channels=hidden_size, out_channels=1) 
      self.tangent = torch.nn.Tanh()

      self.dense = torch.nn.Linear(in_features=2*hidden_size, out_features=1)
      # self.sigmoid = torch.nn.Sigmoid()

      self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=loss_pos_weight) # WeightedBCELoss() # torch.nn.BCELoss()
      self.optimizer = Adam(self.parameters(), lr=lr)
      self.to(self.device)

    def forward(self, x):
        x, _ = self.lstm_1(x)
        x = self.tangent(x)

        x, _ = self.lstm_2(x)
        x = self.tangent(x)

        x = self.dense(x)
        # output = self.sigmoid(x) 
        return x