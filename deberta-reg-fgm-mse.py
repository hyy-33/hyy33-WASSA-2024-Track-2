import torch
from torch import optim, nn
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, AdamW
from transformers import DebertaModel, DebertaTokenizer, DebertaForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
import random
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import PearsonCorrCoef
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import sentencepiece
# import pytorch_warmup as warmup


df = pd.read_csv("trac2_CONVT_train_p.csv")
df_dev = pd.read_csv("trac2_CONVT_dev_total.csv")

predict_target = 'EmotionalPolarity'

# local loading
tokenizer = DebertaTokenizer.from_pretrained('./240601')

# random the order of samples
random.seed(42)
df = df.sample(frac=1).reset_index(drop=True)
df_dev = df_dev.sample(frac=1).reset_index(drop=True)


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class AttentionPooling(nn.Module):
    def __init__(self, in_dim=768):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 1),
        ).to(device)

    def forward(self, last_hidden_state, attention_mask):
        w = self.attention(last_hidden_state).float()
        w[attention_mask == 0] = float('-inf')
        w = torch.softmax(w, 1)
        attention_embeddings = torch.sum(w * last_hidden_state, dim=1)
        return attention_embeddings


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)  # 2-norm
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class Regressor(nn.Module):

    def __init__(self, drop_rate=0.1):
        super(Regressor, self).__init__()
        D_in, D_out = 768, 1

        self.deberta = DebertaModel.from_pretrained('./240601')  # loading pretrained model from local path
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(D_in, D_out))

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids, attention_mask)  # outputs.last_hidden_layer, outputs.pooler_output
        pooling_layer = AttentionPooling()
        pooling_output = pooling_layer(outputs.last_hidden_state.to(device), attention_mask.to(device))
        pred_val = self.regressor(pooling_output)

        return pred_val


model = Regressor()


class Task2Dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx]['text']
        label = self.dataframe.iloc[idx]['EmotionalPolarity']
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length,
                                  return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float32)
        }


# params
bs = 200  # for deberta
learning_rate = 1e-6
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
exp_lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
# warmup_period = 10
# warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=warmup_period)

train_dataset = Task2Dataset(df[:], tokenizer)
dev_dataset = Task2Dataset(df_dev[:], tokenizer)

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=bs, shuffle=False)

# 使用多个GPU
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
pearson = PearsonCorrCoef().to(device)

loss_function = nn.MSELoss()


def evaluate():
    model.eval()
    total_eval_MSELoss = 0
    y_pred = torch.tensor([0.0]).to(device)
    y_truth = torch.tensor([0.0]).to(device)

    for batch in tqdm(dev_loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_function(outputs.float(), labels.reshape(-1, 1).float())
        total_eval_MSELoss += loss.item()

        y_pred = torch.cat((y_pred, outputs.reshape(-1).float()), dim=0)
        y_truth = torch.cat((y_truth, labels.float()), dim=0)

    pear_corr = pearson(y_pred[1:].to(torch.float), y_truth[1:].to(torch.float))
    return total_eval_MSELoss, pear_corr


epochs = 50
res_file = predict_target + "-deberta-reg-fgm-mse epoch=" + str(epochs) + " lr=" + str(learning_rate) + " bs=" + str(bs) + ".txt"
file = open(res_file, "a")
fgm = FGM(model)
for epoch in range(epochs):
    total_train_MSELoss = 0
    y_pred = torch.tensor([0.0]).to(device)
    y_truth = torch.tensor([0.0]).to(device)

    model.train()

    for batch in tqdm(train_loader, desc="Epoch {}".format(epoch + 1)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = loss_function(outputs.float(), labels.reshape(-1, 1).float())
        loss.backward()

        # fgm add:
        fgm.attack()
        outputs = model(input_ids, attention_mask)
        loss_sum = loss_function(outputs.float(), labels.reshape(-1, 1).float())
        loss_sum.backward()  # accumulate gradient of adversarial training
        fgm.restore()  # restore Embedding params

        optimizer.step()
        total_train_MSELoss += loss.item()

        y_pred = torch.cat((y_pred, outputs.reshape(-1).float()), dim=0)
        y_truth = torch.cat((y_truth, labels.float()), dim=0)

    exp_lr_scheduler.step()

    pearson_corr = pearson(y_pred[1:].to(torch.float), y_truth[1:].to(torch.float))
    train_res = "Train: " + str(epoch) + " MSEloss: " + str(total_train_MSELoss) + " pear_corr: " + str(pearson_corr.item())
    print(train_res)
    file.write(train_res + "\n")

    if (epoch + 1) % 2 == 0:  # evaluate
        loss, pear = evaluate()
        eval_res = "Dev: " + str(epoch) + " MSEloss: " + str(loss) + " pear_corr: " + str(pear.item())
        print(eval_res)
        file.write(eval_res + "\n")
        torch.save(model, './deberta-reg-fgm-mse/' + str(pear.item()) + "-" + str(epoch) + '.pth')

file.close()


