import torch
from torch import nn
import torch.nn.functional as F


# Entropic Open-set loss
class EntropicOpenSetLoss():
    def __init__(self, class_names):
        self.class_num = len(class_names)
        self.Cross_entropy = nn.CrossEntropyLoss()

    def __call__(self, output, target_batch):
        output_softmax = F.softmax(output, dim=1)
        loss = 0
        for i, target in enumerate(target_batch):
            if target == self.class_num - 1:
                # background class
                div = 1/(self.class_num-1)
                for index in range(self.class_num - 1):
                    loss -= torch.log(output_softmax[i][index]) * div
            else:
                # CELoss
                loss -= torch.log(output_softmax[i][target])
                # print(loss)

        loss_mean = torch.div(loss,len(target_batch))
        return loss_mean
