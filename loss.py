import torch
from torch import nn
import torch.nn.functional as F


# Entropic Open-set loss

class EntropicOpenSetLoss():
    def __init__(self, class_names):
        self.class_num = len(class_names)

    def __call__(self, output, target_batch):
        output_softmax = F.softmax(output, dim=1)
        div = 1 / (self.class_num - 1)
        loss = 0

        for i, target in enumerate(target_batch):
            if target == self.class_num - 1:
                # background class
                for index in range(self.class_num - 1):
                    loss -= torch.log(output_softmax[i][index]) * div
                    # loss -= torch.log(1 - output_softmax[i][index])
            else:
                # CELoss
                loss -= torch.log(output_softmax[i][target])

        loss_mean = torch.div(loss, len(target_batch))
        return loss_mean

