from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):

    def __init__(self, vocab_size, embedding_size):
        super(SkipGram, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.hidden = nn.Linear(1, self.embedding_size)
        self.output = nn.Linear(1, self.vocab_size)

    def forward(self, x):
        output = self.hidden(x)
        output = self.output(x)
        return output
