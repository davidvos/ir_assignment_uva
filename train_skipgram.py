from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

from SkipGram import SkipGram
from data_processing import APDataset


def train(window_size, embedding_size):
    np.random.seed(0)

    dataset = APDataset(window_size)
    data_generator = dataset.get_generator()

    skip_gram = SkipGram(dataset.vocab_size, embedding_size)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.nn.CrossEntropyLoss()

    loss_list = []

    step = 0

    skip_gram.train()

    for sample, target in data_generator:
        x = np.reshape(sample, (dataset.vocab_size, 1))
        x = torch.tensor(x).float()
        t = np.reshape(target, (dataset.vocab_size, 1))
        t = torch.tensor(t).float()
        y = skip_gram.forward(x).float()
        print(y)
        loss = loss_function(y, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print('Step: ', step, ', Accuracy: ', loss)

        step += 1


if __name__ == '__main__':
    train(2, 100)
