import numpy as np
import gzip
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from transformers import AutoTokenizer
import math

from model import AE


enc = AutoTokenizer.from_pretrained(
  'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',
  bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
)
encode = lambda x: enc.encode(x)
decode = lambda x: enc.decode(x)


def load_dataset(cache_destination):
    f = gzip.GzipFile(cache_destination, "r")
    tokens = np.load(f)
    return tokens

corpus = load_dataset("./dataset/dataset_cache.tar.gz")

kernel_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TokenedDataset(Dataset):
    def __init__(self, corpus, kernel_size):
        self.corpus = corpus
        self.kernel_size = kernel_size

    def __getitem__(self, index):
        return torch.from_numpy(self.corpus[index][1:kernel_size+1]).type(dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.corpus)

corpus_dataset = TokenedDataset(corpus, kernel_size=kernel_size)
train_dataloader = DataLoader(corpus_dataset, batch_size=128, shuffle=True, drop_last=True)

# x_ = next(iter(train_dataloader))

nb_epochs = 5000
learning_rate = 0.001

model = AE([16, 8,8,8, 8, 4]).to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(nb_epochs):
    for i, x in enumerate(train_dataloader):
        pred = model(x)
        loss = nn.functional.mse_loss(pred, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"EPOCH: {epoch}/{nb_epochs} | Loss: {loss.item()}")

torch.save(model, "model.pt")

print(x[0].tolist())
print(pred[0].type(torch.int).cpu().detach().tolist())
