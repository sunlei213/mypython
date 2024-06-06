# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:01:42 2024

@author: sunlei
"""

##################################训练模型，线上用CPU训练会有点久##########################################
############################训练过后应用就不需要在训练了##############################################
#太慢了，训练5个epoch就算了~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim import lr_scheduler
import torch.nn as nn
from tqdm import tqdm
train_data = np.load('train_DATA.npy')
train_label = np.load('train_LABEL.npy')


batch_size = 50
learning_rate = 0.0001
num_epochs = 10
MODEL = 'baseline'

Train_data = torch.tensor(train_data, dtype=torch.float32)
Train_label = torch.tensor(train_label, dtype=torch.float32)

#Val_data = torch.tensor(test_data, dtype=torch.float32)
#Val_label = torch.tensor(test_label, dtype=torch.float32)

dataset_train = TensorDataset(Train_data, Train_label)
#dataset_val = TensorDataset(Val_data, Val_label)

train_loader = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
#val_loader = DataLoader(dataset_val, shuffle=True, batch_size=batch_size)

class model(nn.Module):
    def __init__(self,
                 fc1_size=2000,
                 fc2_size=1000,
                 fc3_size=100,
                 fc1_dropout=0.2,
                 fc2_dropout=0.2,
                 fc3_dropout=0.2,
                 num_of_classes=50):
        super(model, self).__init__()

        self.f_model = nn.Sequential(
            nn.Linear(5088, fc1_size),  # 887
            nn.BatchNorm1d(fc1_size),
            nn.ReLU(),
            nn.Dropout(fc1_dropout),
            nn.Linear(fc1_size, fc2_size),
            nn.BatchNorm1d(fc2_size),
            nn.ReLU(),
            nn.Dropout(fc2_dropout),
            nn.Linear(fc2_size, fc3_size),
            nn.BatchNorm1d(fc3_size),
            nn.ReLU(),
            nn.Dropout(fc3_dropout),
            nn.Linear(fc3_size, 1),

        )

        self.conv_layers1 = nn.Sequential(
            nn.Conv1d(9, 16, kernel_size=1),
            nn.BatchNorm1d(16),
            nn.Dropout(fc3_dropout),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.Dropout(fc3_dropout),
            nn.ReLU(),
        )

        self.conv_2D = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2),
            nn.BatchNorm2d(16),
            nn.Dropout(fc3_dropout),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.BatchNorm2d(32),
            nn.Dropout(fc3_dropout),
            nn.ReLU(),
        )
        hidden_dim = 32
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=4, batch_first=True,
                            # dropout=fc3_dropout,
                            bidirectional=True)
        hidden_dim = 1
        self.l = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=4, batch_first=True,
                         # dropout=fc3_dropout,
                         bidirectional=True)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):

        apply = torch.narrow(x, dim=-1, start=0, length=1).squeeze(1)
        redeem = torch.narrow(x, dim=-1, start=1, length=1).squeeze(1)
        apply, _ = self.l(apply)
        redeem, _ = self.l(redeem)
        apply = torch.reshape(apply, (apply.shape[0], apply.shape[1] * apply.shape[2]))
        redeem = torch.reshape(redeem, (redeem.shape[0], redeem.shape[1] * redeem.shape[2]))

        ZFF = torch.narrow(x, dim=-1, start=2, length=1).squeeze(1)
        HS = torch.narrow(x, dim=-1, start=3, length=1).squeeze(1)
        ZFF, _ = self.l(ZFF)
        HS, _ = self.l(HS)
        ZFF = torch.reshape(ZFF, (ZFF.shape[0], ZFF.shape[1] * ZFF.shape[2]))
        HS = torch.reshape(HS, (HS.shape[0], HS.shape[1] * HS.shape[2]))
        
        xx = x.unsqueeze(1)
        xx = self.conv_2D(xx)
        xx = torch.reshape(xx, (xx.shape[0], xx.shape[1] * xx.shape[2] * xx.shape[3]))
        x = x.transpose(1, 2)
        x = self.conv_layers1(x)
        out = x.transpose(1, 2)
        out2, _ = self.lstm(out)
        out2 = torch.reshape(out2, (out2.shape[0], out2.shape[1] * out2.shape[2]))

        IN = torch.cat((xx, out2, apply, redeem, ZFF, HS), dim=1)
        out = self.f_model(IN)
        return out

model = model()
model.to('cpu')

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
L_train = []
L_val = []
AUC = []
min_validation_loss = 0
for epoch in tqdm(range(num_epochs)):
    PREDICT = []
    TRUE = []
    train_running_loss = 0.0
    counter = 0
    model.train()
    for seq, y in train_loader:
        counter += 1
        output = model(seq.to('cpu'))
        PREDICT.extend(output.detach().cpu().numpy())
        TRUE.extend(y.cpu().numpy())
        loss = criterion(output, y.unsqueeze(1).to('cpu'))
        train_running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    TL = train_running_loss / counter
    L_train.append(TL)
    PP = np.array(PREDICT)
    TT = np.array(TRUE)
    flattened_array1 = PP.flatten()
    flattened_array2 = TT.flatten()
    correlation_matrix = np.corrcoef(flattened_array1, flattened_array2)
    pr_auc = correlation_matrix[0, 1]
    print("Train loss: ", TL, 'correlation_value', pr_auc)
        ################################################################################################################
    if min_validation_loss < pr_auc:
        min_validation_loss = pr_auc
        best_epoch = epoch
        print('Max pr_auc ' + str(min_validation_loss) + ' in epoch ' + str(best_epoch))
        torch.save(model.state_dict(), fr"./model_{MODEL}.pt")
    AUC.append(pr_auc)

