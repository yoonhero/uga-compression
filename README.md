# Text Compression with Various Algorithms

UGAUGA Language proposed by Lee yuchan is so long. I'd like to shorten this using text compression language model.

## 1. Simple Compression

According to pattern frequency, this algorithm compress UGA text.

```bash
# for example
우우가 => 우!
```

## 2. Simple AE

I try to encode and decode this text using simple linear autoencoder. The performance of this model wasn't nice. The reason for this problem may be the tokenizer. 