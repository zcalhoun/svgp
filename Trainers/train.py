"""
This script contains the functions for the main training logic.

"""

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_model(model, likelihood, mll, train_loader, num_epochs, lr):
    """
    The main training function.
    """

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters()},
            {"params": likelihood.parameters()},
        ],
        lr=lr,
    )

    t_max = num_epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, t_max)

    for _ in range(num_epochs):

        for x_batch, y_batch, w_batch in train_loader:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            w_batch = w_batch.cuda() if w_batch is not None else None

            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch, weights=w_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
