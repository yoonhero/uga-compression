import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(self, in_features, out_features, last=False):
        super().__init__()
        self.last = last
        # first simple linear
        self.l1_layer = nn.Linear(in_features, out_features)
        # self.l2_layer = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.l1_layer(x)
        # x = self.relu(x)
        # x = self.l2_layer(x)
        if self.last:
            x = self.sigmoid(x)
        else:
            x = self.relu(x)
            x = self.dropout(x)
        return x


class DecoderBlcok(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.l1_layer = nn.Linear(in_features, out_features)
        # self.l2_layer = nn.Linear(in_features*4, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.l1_layer(x)
        # x = self.relu(x)
        # x = self.l2_layer(x)
        x = self.relu(x)
        

        return x
    

class AE(nn.Module):
    def __init__(self, pattern=[16, 8, 4]):
        super().__init__()

        self.pattern = pattern
        
        self.encoder_blocks = nn.ModuleList([EncoderBlock(in_features, out_features, last=out_features==self.pattern[-1]) for in_features, out_features in zip(self.pattern, self.pattern[1:])])
        print(self.encoder_blocks)

        self.decoder_blocks = nn.ModuleList([DecoderBlcok(in_features, out_features) for in_features, out_features in zip(self.pattern[::-1], self.pattern[::-1][1:])])

    def encode(self, x):
        for e in self.encoder_blocks:
            x = e(x)
        return x

    def decode(self, x):
        for d in self.decoder_blocks:
            x = d(x)
        return x

    def forward(self, x):
        encoded_vector = self.encode(x)
        decoded_vector = self.decode(encoded_vector)

        return decoded_vector   