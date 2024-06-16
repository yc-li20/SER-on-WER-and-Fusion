import torch
import numpy as np
import torch.nn as nn
import os, random, re
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.metrics import accuracy_score, confusion_matrix


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

bert_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
bert_model = RobertaModel.from_pretrained('roberta-base').to(device)


def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d{2}:\d{2}:\d{2}', '', text)
    text = text.replace('\n', ' ')
    return text


names= []
acc_all = []
emolabels = []
bertfeats = torch.empty(0, 768)  

with open('your_prepared_transcript', 'r') as f:
    files = f.readlines()
    for line in files:
        emotion, file_name, utterance = map(str.strip, line.split(","))
        utterance = clean_text(utterance)
        emotion_label = {"Ang": 0, "Hap": 1, "Neu": 2, "Sad": 3}.get(emotion)
        
        bert_inputs = bert_tokenizer(utterance.lower(), return_tensors="pt").to(device)
        bert_outputs = bert_model(**bert_inputs).last_hidden_state.mean(dim=1).cpu()
        
        names.append(file_name)
        emolabels.append(emotion_label)
        bertfeats = torch.cat((bertfeats, bert_outputs.data), 0)
            
        torch.cuda.empty_cache()


class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.dense1 = nn.Linear(768, 128)
        self.flat = nn.Flatten()
        self.dense = nn.Linear(128, 16)
        self.acti = nn.ReLU()
        self.out = nn.Linear(16, 4)

    def forward(self, x):
        x = self.flat(x)
        x = self.dense1(x)
        x = self.acti(x)
        x = self.dense(x)
        res = self.acti(x)
        emotion = self.out(res)
        return emotion


for session in ['Ses01', 'Ses02', 'Ses03', 'Ses04', 'Ses05']:
    
    names_train = [names[i] for i in range(len(names)) if session not in names[i]]
    names_test = [names[i] for i in range(len(names)) if session in names[i]]

    emolabels_train = [emolabels[i] for i in range(len(names)) if session not in names[i]]
    emolabels_test = [emolabels[i] for i in range(len(names)) if session in names[i]]

    bert_train = torch.stack([bertfeats[i] for i in range(len(names)) if session not in names[i]], dim=0)
    bert_test = torch.stack([bertfeats[i] for i in range(len(names)) if session in names[i]], dim=0)

    bert_train = bert_train.view(len(names_train), 1, 768)
    bert_test = bert_test.view(len(names_test), 1, 768)

    train_dataset = MyDataset(bert_train, emolabels_train)
    test_dataset = MyDataset(bert_test, emolabels_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = NeuralNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, eps=1e-8, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()


    batch_size = 64
    best_acc = 0.0
    num_epochs = 100


    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        predictions_train = []

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            predictions_train.extend(torch.argmax(outputs, dim=1).cpu().numpy())

        model.eval()
        test_loss = 0.0
        predictions_test = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs.to(device))
                loss = criterion(outputs, labels.to(device))
                test_loss += loss.item()
                predictions_test.extend(torch.argmax(outputs, dim=1).cpu().numpy())

        acc_train = accuracy_score(emolabels_train, predictions_train)
        acc_test = accuracy_score(emolabels_test, predictions_test)

        if acc_test > best_acc:
            best_acc = acc_test

    print(f'{session} Best Test Accuracy: {best_acc:.4f}')
    acc_all.append(best_acc)

acc_final = sum(acc_all) / len(acc_all)
print(f'Final Accuracy: {acc_final:.4f}')
