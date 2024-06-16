import torch
import numpy as np
import torch.nn as nn
import os, random, re, pickle
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True


with open('Train.txt', 'r') as f:
    train_names = [line.strip().split(',')[0] for line in f.readlines()]
with open('Dev.txt', 'r') as f:
    val_names = [line.strip().split(',')[0] for line in f.readlines()]
with open('Test1.txt', 'r') as f:
    test1_names = [line.strip().split(',')[0] for line in f.readlines()]



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.flat = nn.Flatten()
        self.dense1 = nn.Linear(768, 128)
        self.dense = nn.Linear(128, 16)
        self.acti = nn.ReLU()
        self.out_a = nn.Linear(16, 1)
        self.out_v = nn.Linear(16, 1)
        self.out_d = nn.Linear(16, 1)

    def forward(self, x):
        x = self.flat(x)
        x = self.dense1(x)
        x = self.acti(x)
        x = self.dense(x)
        res = self.acti(x)
        arousal = self.out_a(res).squeeze(1)
        valence = self.out_v(res).squeeze(1)
        dominance = self.out_d(res).squeeze(1)
        return arousal, valence, dominance

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d{2}:\d{2}:\d{2}', '', text)
    text = text.replace('\n', ' ')
    return text


class CCC_loss(nn.Module):
    def __init__(self):
        super(CCC_loss, self).__init__()

    def forward(self, pred, lab):

        m_pred = torch.mean(pred, 0, keepdim=True)
        m_lab = torch.mean(lab, 0, keepdim=True)

        d_pred = pred - m_pred
        d_lab = lab - m_lab

        v_pred = torch.var(pred, 0, unbiased=False)
        v_lab = torch.var(lab, 0, unbiased=False)

        corr = torch.sum(d_pred * d_lab, 0) / (torch.sqrt(torch.sum(d_pred ** 2, 0)) * torch.sqrt(torch.sum(d_lab ** 2, 0)))

        s_pred = torch.std(pred, 0, unbiased=False)
        s_lab = torch.std(lab, 0, unbiased=False)

        ccc = (2*corr*s_pred*s_lab) / (v_pred + v_lab + (m_pred[0]-m_lab[0])**2)    
        return ccc


train_labels, val_labels, test1_labels = [], [], []
for i in range(len(a_train)):
    train_labels.append([a_train[i], v_train[i], d_train[i]])
for i in range(len(a_val)):
    val_labels.append([a_val[i], v_val[i], d_val[i]])
for i in range(len(a_test1)):
    test1_labels.append([a_test1[i], v_test1[i], d_test1[i]])


train_dataset = MyDataset(bert_train, train_labels)
val_dataset = MyDataset(bert_val, val_labels)
test1_dataset = MyDataset(bert_test1, test1_labels)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test1_loader = DataLoader(test1_dataset, batch_size=batch_size, shuffle=False)


model = NeuralNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, eps=1e-8, weight_decay=1e-5)
criterion = CCC_loss()


num_epochs = 30
best_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs_a, outputs_v, outputs_d = model(inputs.to(device))
        loss_a = 1.0 - criterion(outputs_a, labels[:, 0].to(device))
        loss_v = 1.0 - criterion(outputs_v, labels[:, 1].to(device))
        loss_d = 1.0 - criterion(outputs_d, labels[:, 2].to(device))
        loss = loss_a + loss_v + loss_d
        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs_a, outputs_v, outputs_d = model(inputs.to(device))
            loss_a = 1.0 - criterion(outputs_a, labels[:, 0].to(device))
            loss_v = 1.0 - criterion(outputs_v, labels[:, 1].to(device))
            loss_d = 1.0 - criterion(outputs_d, labels[:, 2].to(device))
            loss = loss_a + loss_v + loss_d
            loss = torch.mean(loss)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    
    print(f'Epoch: {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
    
    if val_loss < best_loss:
        best_loss = val_loss
        best_model_params = model.state_dict().copy()

model.load_state_dict(best_model_params)

model.eval()
a_predictions_test1 = []
v_predictions_test1 = []
d_predictions_test1 = []

with torch.no_grad():
    for inputs, labels in test1_loader:
        outputs_a, outputs_v, outputs_d = model(inputs.to(device))
        a_predictions_test1.extend(outputs_a.detach())
        v_predictions_test1.extend(outputs_v.detach())
        d_predictions_test1.extend(outputs_d.detach())

ccc_a_test1 = criterion(torch.tensor(a_predictions_test1).to(device), torch.tensor(a_test1).to(device))
ccc_v_test1 = criterion(torch.tensor(v_predictions_test1).to(device), torch.tensor(v_test1).to(device))
ccc_d_test1 = criterion(torch.tensor(d_predictions_test1).to(device), torch.tensor(d_test1).to(device))

print(f'CCC_a_test1: {ccc_a_test1:.4f} | CCC_v_test1: {ccc_v_test1:.4f} | CCC_d_test1: {ccc_d_test1:.4f}')