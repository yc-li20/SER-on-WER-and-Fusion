import numpy as np
import torch, pickle
import torch.nn as nn
import os, random, re
from transformers import RobertaTokenizer, RobertaModel
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True


with open('train_name.txt', 'r') as f:
    train_names = [line.strip() for line in f.readlines()]
with open('val_name.txt', 'r') as f:
    val_names = [line.strip() for line in f.readlines()]
with open('test_name.txt', 'r') as f:
    test_names = [line.strip() for line in f.readlines()]
    

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d{2}:\d{2}:\d{2}', '', text)
    text = text.replace('\n', ' ')
    return text


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bert_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
bert_model = RobertaModel.from_pretrained('roberta-base').to(device)


    
names_train, names_val, names_test = [], [], []
emolabels_train, emolabels_val, emolabels_test = [], [], []
bert_train, bert_val, bert_test = torch.empty(0, 768), torch.empty(0, 768), torch.empty(0, 768)


with open('your_prepared_transcript', 'r') as f:
    files = f.readlines()
    for line in files:
        file_name, utterance, emotion = map(str.strip, line.split(","))
        utterance = clean_text(utterance)

        emotion_label = float(emotion)

        bert_inputs = bert_tokenizer(utterance.lower(), return_tensors="pt").to(device)
        bert_outputs = bert_model(**bert_inputs).last_hidden_state.mean(dim=1).cpu()

        if file_name in train_names:
            names_train.append(file_name)
            emolabels_train.append(emotion_label)
            bert_train = torch.cat((bert_train, bert_outputs.data), 0)

        elif file_name in val_names:
            names_val.append(file_name)
            emolabels_val.append(emotion_label)
            bert_val = torch.cat((bert_val, bert_outputs.data), 0)

        elif file_name in test_names:
            names_test.append(file_name)
            emolabels_test.append(emotion_label)
            bert_test = torch.cat((bert_test, bert_outputs.data), 0)

        torch.cuda.empty_cache()

bert_train = bert_train.view(len(names_train), 1, 768)
bert_val = bert_val.view(len(names_val), 1, 768)
bert_test = bert_test.view(len(names_test), 1, 768)


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
        self.dense1 = nn.Linear(768, 128)
        self.flat = nn.Flatten()
        self.dense = nn.Linear(128, 16)
        self.acti = nn.Sigmoid()
        self.out = nn.Linear(16, 1)

    def forward(self, x):
        x = self.flat(x)
        x = self.dense1(x)
        x = self.acti(x)
        x = self.dense(x)
        res = self.acti(x)
        emotion = self.out(x).squeeze(1)
        return emotion


def multiclass_acc(truths, preds):
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


train_dataset = MyDataset(bert_train, emolabels_train)
val_dataset = MyDataset(bert_val, emolabels_val)
test_dataset = MyDataset(bert_test, emolabels_test)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = NeuralNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, eps=1e-8, weight_decay=1e-5)
criterion = nn.MSELoss()

best_acc = 0.0
best_loss = 100.0
num_epochs = 100
best_model_params = None

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
        predictions_train.extend(outputs.detach().cpu().numpy())

    model.eval()
    val_loss = 0.0
    predictions_val = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs.to(device)).view(-1)
            loss = criterion(outputs, labels.to(device))
            val_loss += loss.item()
            predictions_val.extend(outputs.detach().cpu().numpy())

        val_loss /= len(val_loader)

    if val_loss < best_loss:
        best_loss = val_loss
        best_model_params = model.state_dict().copy()


model.load_state_dict(best_model_params)

test_loss = 0.0
predictions_test = []

model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs.to(device)).view(-1)
        loss = criterion(outputs, labels.to(device))
        predictions_test.extend(outputs.detach().cpu().numpy())

non_zeros_test = np.array([i for i, e in enumerate(emolabels_test) if e != 0])
non_zeros_binary_truth_test = (np.array(emolabels_test)[non_zeros_test] > 0)
non_zeros_binary_preds_test = (np.array(predictions_test)[non_zeros_test] > 0)

non_zeros_acc2_test = accuracy_score(non_zeros_binary_truth_test, non_zeros_binary_preds_test)

binary_truth_test = (np.array(emolabels_test) >= 0)
binary_preds_test = (np.array(predictions_test) >= 0)
acc2_test = accuracy_score(binary_truth_test, binary_preds_test)

test_truth_a7 = np.clip(emolabels_test, a_min=-3., a_max=3.)
test_preds_a7 = np.clip(predictions_test, a_min=-3., a_max=3.)
acc7_test = multiclass_acc(test_truth_a7, test_preds_a7)

mae_test = mean_absolute_error(emolabels_test, predictions_test)

print(f'Neg/Non-Neg: {acc2_test:.4f}, Pos/Neg: {non_zeros_acc2_test:.4f}, acc7: {acc7_test:.4f}, mae: {mae_test:.4f}')
