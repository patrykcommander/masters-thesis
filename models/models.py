import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from models.datasets import ECGDataset
from sklearn.metrics import f1_score, confusion_matrix, balanced_accuracy_score
from models.st_res_blocks import ST, ResPath, UpSampleBlock
from models.losses import WeightedBCELoss
from models.transformer_blocks import TransformerEncoderLayer
from customLib.peak_detection import correct_prediction_according_to_aami

class BasicModel(nn.Module):
    def __init__(self, apply_sigmoid=False, checkpoint_path=None, name=None):
        super(BasicModel, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Training will be performed with:', self.device)
        self.to(self.device)
        
        self.best_acc = 0.0
        self.apply_sigmoid = apply_sigmoid

        self.checkpoint_path = checkpoint_path
        self.name = name

        self.metrics = {
            "train": {
                "loss": [], 
                "f1": [], 
                "accuracy": []
            }, 
            "validation": {
                "loss": [], 
                "f1": [], 
                "accuracy": []
            }, 
            "test": {
                "loss": [], 
                "f1": [], 
                "accuracy": []
            }
        }

    def forward(self, x):
        pass

    def on_epoch_end(self, epoch, accuracy, f1):
        if accuracy > self.best_acc:
            self.best_acc = accuracy
            torch.save(self.state_dict(), f"{self.checkpoint_path}/{self.name}_epoch_{epoch+1}_acc_{accuracy*100:.2f}_f1_{f1:.2f}.pt")

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

            if self.scheduler is not None:
                self.scheduler.step()
                print("Learning rate: ", self.scheduler.get_last_lr())

            all_outputs = np.array(all_outputs)
            all_labels = np.array(all_labels)
            y_pred_binary = (all_outputs > 0.5).astype(int)
            test_loss = running_loss / len(train_loader)

            print(f"\nTrain Loss: {test_loss:.4f}")
            acc, f1 = self.calculate_metrics(test_loss, all_labels, y_pred_binary, phase="train")
        
            if x_val is not None:
                self.validate(validation_loader)

            self.on_epoch_end(epoch, accuracy=acc, f1=f1)
        
        with open(f"./metrics/{self.name}.pkl", "wb") as f: # save metrics after training
            pickle.dump(self.get_metrics(), f)

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
            validation_loss = running_vloss / len(validation_loader)

            print(f"\nValidation Loss: {validation_loss:.4f}")
            self.calculate_metrics(validation_loss, all_labels, y_pred_binary, phase="validation")
    
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

                if plot and (i % int((len(test_loader) / 10)) == 0):
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
        test_loss = running_loss / len(test_loader)


        print(f"\nTest Loss: {test_loss:.4f}")
        self.calculate_metrics(test_loss, all_labels, y_pred_binary, phase="test")
    
    # we only care about the precision of the R_peaks (binary class 1) and we about the false positive rate
    def calculate_metrics(self, loss, y_true, y_pred_binary, phase="train"):
        # R-wave prediction in the particular neighbourhood of the labeled sample treated as correct
        # according to the AAMI standard, the R-peak prediction is considered to be correct (TP) 
        # if its time deviation from each side of the real R-peak position is less than 75 ms. Should this time difference be greater,
        # the R-peak is considered to be a false positive

        total_targets = y_true.shape[0]
        positive_count = np.sum(y_true)
        negative_count = total_targets - positive_count
        w_p = negative_count / total_targets
        w_n = positive_count / total_targets

        weights = [ w_p if x == 1 else w_n for x in y_true ]

        y_pred_binary = correct_prediction_according_to_aami(y_true, y_pred_binary)

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

        self.metrics[phase]["loss"].append(round(loss, 5))
        self.metrics[phase]["f1"].append(round(f1, 5))
        self.metrics[phase]["accuracy"].append(round(accuracy, 5))

        return accuracy, f1
    
    def get_metrics(self):
        return self.metrics

class ST_RES_NET(BasicModel):
    def __init__(self, learning_rate=1e-4, loss_pos_weight=None, loss_neg_weight=None):
        super(ST_RES_NET, self).__init__(apply_sigmoid=False, checkpoint_path="./checkpoints/st_res_net", name="ST_RES_NET")
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

        self.criterion = WeightedBCELoss(positive_weight=loss_pos_weight, negative_weight=loss_neg_weight) # nn.BCELoss()
        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        self.scheduler = None

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
    def __init__(self, input_dim, hidden_size, lr=1e-2, loss_pos_weight=None, checkpoint_path="./checkpoints/lstm", name="lstm"):
      super(LSTM, self).__init__(apply_sigmoid=True,  name=name, checkpoint_path=checkpoint_path)
      self.lstm_1 = torch.nn.LSTM(input_size=input_dim, hidden_size=hidden_size, batch_first=True, bidirectional=True, dropout=0.3)
      self.lstm_2 = torch.nn.LSTM(input_size=2*hidden_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)
      #self.relu = torch.nn.ReLU()
      # achieveing better results with Dense layer instead of Conv1d and when dropout is used after lstm instead of Dense layer
      #self.conv = torch.nn.Conv1d(kernel_size=1, in_channels=hidden_size, out_channels=1) 
      self.tangent = torch.nn.Tanh()

      self.dense = torch.nn.Linear(in_features=2*hidden_size, out_features=1)
      # self.sigmoid = torch.nn.Sigmoid()

      self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=loss_pos_weight) # WeightedBCELoss() # torch.nn.BCELoss()
      self.optimizer = Adam(self.parameters(), lr=lr, amsgrad=True)
      self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[15,25], gamma=0.1)

      self.to(self.device)

    def forward(self, x):
        x, _ = self.lstm_1(x)
        x = self.tangent(x)

        x, _ = self.lstm_2(x)
        x = self.tangent(x)

        x = self.dense(x)
        # output = self.sigmoid(x) 
        return x

class TransRR(BasicModel):
    def __init__(self, num_layers, input_dim, nhead, dropout, loss_pos_weight, loss_neg_weight, learning_rate=1e-4):
        super(TransRR, self).__init__(name="transRR", checkpoint_path="./checkpoints/transRR")

        self.embedding_layer = nn.Sequential(
            nn.Conv1d(1, 16, padding=1, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, padding=1, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(32, input_dim, padding=1, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(64, input_dim, padding=1, kernel_size=3),
            nn.ReLU(),
        )

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(input_dim=input_dim, nhead=nhead, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

        self.linear_1 = nn.Linear(input_dim, 1)

        self.sigmoid = nn.Sigmoid()

        self.criterion = WeightedBCELoss(positive_weight=loss_pos_weight, negative_weight=loss_neg_weight) # nn.BCELoss()
        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        self.to(self.device)


    def forward(self, src):
        src = self.embedding_layer(src)

        output = src.permute(0,2,1)
        for layer in self.layers:
            output = layer(output)
 
        output = self.linear_1(output)
        
        return self.sigmoid(output)

class SimpleTransformerModel(BasicModel):
    def __init__(self, input_dim, seq_length, num_layers, num_heads, dim_feedforward, dropout):
        super(SimpleTransformerModel, self).__init__(name="TNET", checkpoint_path="./checkpoints/TNET")
        
        self.input_dim = input_dim
        self.seq_length = seq_length

        self.embedding_layer = nn.Sequential(
            nn.Conv1d(1, 16, padding=1, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, padding=1, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(32, input_dim, padding=1, kernel_size=3),
            nn.ReLU(),
        )
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.dense = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self.criterion = WeightedBCELoss(negative_weight=1, positive_weight=10)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.scheduler = None

        self.to(self.device)

    def forward(self, x):
        # x shape: [batch_size, seq_length, 1]
        x = x.permute(0, 2, 1)  # Reshape to [batch_size, in_channels (1), seq_length]
        embedded = self.embedding_layer(x)  # Shape: [batch_size, input_dim, seq_length]
        embedded = embedded.permute(2, 0, 1) # Shape: [seq_length, batch_size, input_dim]

        transformed = self.transformer_encoder(embedded)  # Shape: [seq_length, batch_size, input_dim]
        transformed = transformed.permute(1, 0, 2)  # Reshape to [batch_size, seq_length, input_dim]

        output = self.dense(transformed)  # Shape: [batch_size, seq_length, 1]
        
        return self.sigmoid(output)
       