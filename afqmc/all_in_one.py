'''
from:
https://transformers.run/intro/2021-12-17-transformers-note-4/

code github: https://github.com/jsksxs360/How-to-use-Transformers/tree/main/src/pairwise_cls_similarity_afqmc
'''

import random
import os
import numpy as np
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig
from transformers import BertPreTrainedModel, BertModel
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm


def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
seed_everything(42)

learning_rate = 1e-5
batch_size = 4
epoch_num = 3

checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


class AFQMC(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt') as f:
            for idx, line in enumerate(f):
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# train_data = AFQMC('data/afqmc_public/train.json')
# valid_data = AFQMC('data/afqmc_public/dev.json')

train_data = AFQMC('/root/huggingface_study/afqmc_public/train.json')
valid_data = AFQMC('/root/huggingface_study/afqmc_public/dev.json')


def collote_fn(batch_samples):
    batch_sentence_1, batch_sentence_2 = [], []
    batch_label = []
    for sample in batch_samples:
        batch_sentence_1.append(sample['sentence1'])
        batch_sentence_2.append(sample['sentence2'])
        batch_label.append(int(sample['label']))
    X = tokenizer(
        batch_sentence_1,
        batch_sentence_2,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    y = torch.tensor(batch_label)
    return X, y


train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=collote_fn)


class BertForPairwiseCLS(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(768, 2)
        self.post_init()

    def forward(self, x):
        outputs = self.bert(**x)
        cls_vectors = outputs.last_hidden_state[:, 0, :]
        cls_vectors = self.dropout(cls_vectors)
        logits = self.classifier(cls_vectors)
        return logits


config = AutoConfig.from_pretrained(checkpoint)
model = BertForPairwiseCLS.from_pretrained(checkpoint, config=config).to(device)


def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_step_num = (epoch - 1) * len(dataloader)

    model.train()
    for step, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss / (finish_step_num + step):>7f}')
        progress_bar.update(1)
    return total_loss


def test_loop(dataloader, model, mode='Test'):
    assert mode in ['Valid', 'Test']
    size = len(dataloader.dataset)
    correct = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    correct /= size
    print(f"{mode} Accuracy: {(100 * correct):>0.1f}%\n")
    return correct


# loss_fn = nn.CrossEntropyLoss()
# optimizer = AdamW(model.parameters(), lr=learning_rate)
# lr_scheduler = get_scheduler(
#     "linear",
#     optimizer=optimizer,
#     num_warmup_steps=0,
#     num_training_steps=epoch_num * len(train_dataloader),
# )
#
# total_loss = 0.
# best_acc = 0.
# for t in range(epoch_num):
#     print(f"Epoch {t + 1}/{epoch_num}\n-------------------------------")
#     total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t + 1, total_loss)
#     valid_acc = test_loop(valid_dataloader, model, mode='Valid')
#     if valid_acc > best_acc:
#         best_acc = valid_acc
#         print('saving new weights...\n')
#         torch.save(model.state_dict(), f'epoch_{t + 1}_valid_acc_{(100 * valid_acc):0.1f}_model_weights.bin')
# print("Done!")

'''
Epoch 1/3
-------------------------------
loss: 0.562499: 100%|██████████████████████████████████████████████████████████████████████| 8584/8584 [12:03<00:00, 11.87it/s]
Valid Accuracy: 72.6%

saving new weights...

Epoch 2/3
-------------------------------
loss: 0.510235: 100%|██████████████████████████████████████████████████████████████████████| 8584/8584 [11:11<00:00, 12.79it/s]
Valid Accuracy: 73.4%

saving new weights...

Epoch 3/3
-------------------------------
loss: 0.458185: 100%|██████████████████████████████████████████████████████████████████████| 8584/8584 [11:27<00:00, 12.49it/s]
Valid Accuracy: 73.9%

saving new weights...

Done!

(p3106) [root@k8s-master afqmc]# ll
total 1191820
-rw-r--r--. 1 root root      6398 Aug 18 16:51 all_in_one.py
-rw-r--r--. 1 root root      4627 Aug 18 15:17 data.py
-rw-r--r--. 1 root root 406800696 Aug 18 16:15 epoch_1_valid_acc_72.6_model_weights.bin
-rw-r--r--. 1 root root 406800696 Aug 18 16:27 epoch_2_valid_acc_73.4_model_weights.bin
-rw-r--r--. 1 root root 406800696 Aug 18 16:38 epoch_3_valid_acc_73.9_model_weights.bin
'''

model.load_state_dict(torch.load('epoch_3_valid_acc_73.9_model_weights.bin'))
test_loop(valid_dataloader, model, mode='Test')