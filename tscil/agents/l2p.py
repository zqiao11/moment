from tscil.agents.base import BaseLearner
import numpy as np
import torch
import torch.nn as nn
from tscil.utils.data import extract_samples_according_to_labels, Dataloader_from_numpy
import torch.nn.functional as F


class L2P(BaseLearner):
    def __init__(self, model, args):
        super(L2P, self).__init__(model, args)

    def train_epoch(self, dataloader, epoch):
        total = 0
        correct = 0
        epoch_loss = 0

        self.model.train()

        for batch_id, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            x = x.permute(0, 2, 1)  # Moment takes in tensor of shape [batchsize, n_channels, context_length
            total += y.size(0)
            if y.size == 1:
                y.unsqueeze()

            self.optimizer.zero_grad()
            outputs = self.model(x, reduction=self.args.reduction)
            logits = outputs.logits
            # See https://github.com/JH-LEE-KR/l2p-pytorch/blob/main/engine.py#L60
            classes_to_mask = np.setdiff1d(np.arange(logits.size(1)), self.classes_in_task)
            classes_to_mask = torch.tensor(classes_to_mask, dtype=torch.int64).to(logits.device)
            logits = logits.index_fill(dim=1, index=classes_to_mask, value=float('-inf'))
            step_loss = self.criterion(logits, y)

            if self.args.pull_constraint and outputs.reduce_sim is not None:
                step_loss = step_loss - self.args.pull_constraint_coeff * outputs.reduce_sim

            step_loss.backward()
            self.optimizer_step(epoch)

            epoch_loss += step_loss
            prediction = torch.argmax(logits, dim=1)
            correct += prediction.eq(y).sum().item()

        epoch_acc = 100. * (correct / total)
        epoch_loss /= (batch_id + 1)  # avg loss of a mini batch

        return epoch_loss, epoch_acc

    @torch.no_grad()
    def after_task(self, x_train, y_train):
        self.learned_classes += self.classes_in_task
        if self.update_model:
            self.model.load_state_dict(torch.load(self.ckpt_path))

        if self.use_prototype:
            for i in self.classes_in_task:
                X_i, Y_i = extract_samples_according_to_labels(x_train, y_train, target_ids=[i])
                dataloader = Dataloader_from_numpy(X_i, Y_i, batch_size=self.args.batch_size, shuffle=False)
                features = []
                for (x, y) in dataloader:
                    x = x.to(self.device)
                    x = x.permute(0, 2, 1)  # Moment takes in tensor of shape [batchsize, n_channels, context_length
                    outputs = self.model(x, reduction=self.args.reduction)
                    embeddings = outputs.embeddings
                    features.append(torch.mean(embeddings, dim=1).detach())
                features = torch.cat(features)
                mu = features.mean(0)
                if self.prototype is None:
                    # If prototype is None, initialize it with mu
                    self.prototype = mu.unsqueeze(0)  # Add a new dimension to make it 2D
                else:
                    # If prototype is a tensor, concatenate mu along the appropriate dimension
                    self.prototype = torch.cat((self.prototype, mu.unsqueeze(0)), dim=0)