import torch
from torch import optim, nn
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import DebertaTokenizer, DebertaForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
import random
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torchmetrics import PearsonCorrCoef
import sentencepiece
# import pytorch_warmup as warmup


def create_hierarchical_penalty_matrix(num_classes, thegma):
    """
    Create a matrix of size num_classes x num_classes where each entry (i, j)
    contains the hierarchical penalty for predicting class j when the true class is i.
    """
    # Generate a grid of label indices
    indices = torch.arange(num_classes).unsqueeze(0)
    # Calculate the absolute difference between indices and transpose
    absolute_differences = torch.abs(indices - indices.T)
    # Calculate the hierarchical penalty matrix
    penalty_matrix = torch.exp(-thegma * absolute_differences)
    return penalty_matrix


class CombinedLoss(nn.Module):
    def __init__(self, num_classes, alpha, beta, thegma):
        super(CombinedLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta  # beta param: adjust weights of 2 losses
        self.thegma = thegma
        # calc penalty matrix in advance
        self.penalty_matrix = create_hierarchical_penalty_matrix(num_classes, thegma)

    def forward(self, logits, targets):
        # penalty matrix and logits on the same device
        self.penalty_matrix = self.penalty_matrix.to(logits.device)

        # calc cross-entropy loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # penalty according to real labels of each prediction
        penalties = self.penalty_matrix[targets, :]

        # penalty -> log
        log_probs = F.log_softmax(logits, dim=1)
        weighted_log_probs = penalties * log_probs

        # calc final weighted log probability loss
        structured_contrastive_loss = -torch.sum(weighted_log_probs, dim=1).mean()

        # calc Pearson related loss
        logits_flat = logits.view(-1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        targets_flat = targets_one_hot.view(-1)

        logits_mean = logits_flat.mean()
        targets_mean = targets_flat.mean()

        logits_centered = logits_flat - logits_mean
        targets_centered = targets_flat - targets_mean

        correlation = torch.sum(logits_centered * targets_centered) / (
                    torch.sqrt(torch.sum(logits_centered ** 2)) * torch.sqrt(torch.sum(targets_centered ** 2)))
        pearson_loss = -correlation

        # combine 2 losses
        combined_loss = self.alpha * structured_contrastive_loss + self.beta * pearson_loss

        return combined_loss


df = pd.read_csv("trac2_CONVT_train_p.csv")
df_dev = pd.read_csv("trac2_CONVT_dev_total.csv")

# divide original labels into categories
emotion_bins = [-0.25, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.5, 4.5, 5.5]
emotion_groups = [0, 1, 2, 3, 4, 5, 6, 7, 8]

emotionalPolarity_bins = [-0.25, 0.25, 0.75, 1.25, 1.75, 3]
emotionalPolarity_groups = [0, 1, 2, 3, 4]

empathy_bins = [-0.25, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.5]
empathy_groups = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

predict_target = 'EmotionalPolarityClass'
label_target = emotionalPolarity_groups


# turn original labels into categories
def value_to_class(bins, groups, df_name, column_name):
    class_col = column_name + "Class"
    df_name[class_col] = pd.cut(df_name[column_name], bins, labels=groups)


value_to_class(emotion_bins, emotion_groups, df, 'Emotion')
value_to_class(emotionalPolarity_bins, emotionalPolarity_groups, df, 'EmotionalPolarity')
value_to_class(empathy_bins, empathy_groups, df, 'Empathy')

value_to_class(emotion_bins, emotion_groups, df_dev, 'Emotion')
value_to_class(emotionalPolarity_bins, emotionalPolarity_groups, df_dev, 'EmotionalPolarity')
value_to_class(empathy_bins, empathy_groups, df_dev, 'Empathy')

# loading pretrained models from local path
tokenizer = DebertaTokenizer.from_pretrained('./240601')
model = DebertaForSequenceClassification.from_pretrained('./240601', num_labels=len(label_target))

# random the order of samples
random.seed(42)
df = df.sample(frac=1).reset_index(drop=True)
df_dev = df_dev.sample(frac=1).reset_index(drop=True)


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)  # default: 2-norm
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class Task2Dataset(Dataset):  # dataset
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx]['text']
        label = self.dataframe.iloc[idx][predict_target]
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length,
                                  return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
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

# loss_func = torch.nn.CrossEntropyLoss()
loss_func = CombinedLoss(num_classes=len(label_target), alpha=0.8, beta=0.2, thegma=0.8)
loss_func_fgm = nn.CrossEntropyLoss()
criterion = torch.nn.CrossEntropyLoss()


def evaluate():
    model.eval()
    total_eval_accuracy = 0
    total_eval_CELoss = 0
    y_pred = torch.tensor([0]).to(device)
    y_truth = torch.tensor([0]).to(device)

    for batch in tqdm(dev_loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        ce_loss = criterion(logits, labels)
        total_eval_CELoss += ce_loss.item()

        preds = torch.argmax(logits, dim=1)
        y_pred = torch.cat((y_pred, preds), dim=0)
        y_truth = torch.cat((y_truth, labels), dim=0)

        accuracy = (preds == labels).float().mean()
        total_eval_accuracy += accuracy.item()

        comb_loss = loss_func(logits, labels)

    pearson_corr = pearson(y_pred[1:].to(torch.float), y_truth[1:].to(torch.float))
    average_eval_accuracy = total_eval_accuracy / len(dev_loader)
    return comb_loss.item(), average_eval_accuracy, total_eval_CELoss, pearson_corr.item()


epochs = 50
res_file = predict_target + "-deberta-class-fgm-comb epoch=" + str(epochs) + " lr=" + str(learning_rate) + " bs=" + str(bs) + ".txt"
fgm = FGM(model)
for epoch in range(epochs):
    y_pred = torch.tensor([0]).to(device)
    y_truth = torch.tensor([0]).to(device)

    model.train()
    total_loss = 0
    total_train_CELoss = 0
    total_eval_accuracy = 0

    for batch in tqdm(train_loader, desc="Epoch {}".format(epoch + 1)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        logits = outputs.logits
        loss = loss_func(logits, labels)
        total_loss += loss.item()
        loss.backward()

        preds = torch.argmax(logits, dim=1)
        y_pred = torch.cat((y_pred, preds), dim=0)
        y_truth = torch.cat((y_truth, labels), dim=0)

        accuracy = (preds == labels).float().mean()
        total_eval_accuracy += accuracy.item()

        ce_loss = criterion(logits, labels)
        total_train_CELoss += ce_loss

        # fgm add:
        fgm.attack()
        outputs = model(input_ids, attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        loss_sum = loss_func_fgm(logits, labels)
        loss_sum.backward()  # accumulate gradient of adversarial training
        fgm.restore()  # restore Embedding params

        optimizer.step()

    # with warmup_scheduler.dampening():
    #     if warmup_scheduler.last_step + 1 >= warmup_period:
    #         exp_lr_scheduler.step()
    exp_lr_scheduler.step()

    average_eval_accuracy = total_eval_accuracy / len(train_loader)
    pearson_corr = pearson(y_pred[1:].to(torch.float), y_truth[1:].to(torch.float))

    train_res = "Train: " + str(epoch) + " comb_loss: " + str(total_loss) + " accu: " + str(average_eval_accuracy) + " ce: " + str(total_train_CELoss.item()) + " pear: " + str(pearson_corr.item())
    print(train_res)
    file = open(res_file, "a")
    file.write(train_res + "\n")
    file.close()

    if (epoch + 1) % 2 == 0:  # evaluate
        eval_comb, eval_accu, eval_ce, pear_corr = evaluate()
        eval_res = "Dev: " + str(epoch) + " comb_loss: " + str(eval_comb) + " accu: " + str(eval_accu) + " ce: " + str(eval_ce) + " pear: " + str(pear_corr)
        print(eval_res)
        file = open(res_file, "a")
        file.write(eval_res + "\n")
        file.close()
        torch.save(model, './deberta-class-fgm-comb/' + str(pear_corr) + "-" + str(epoch) + '.pth')
