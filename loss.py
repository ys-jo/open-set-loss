import torch
from torch import nn
import torch.nn.functional as F
import math
# Entropic Open-set loss
class EntropicOpenSetLoss():
    def __init__(self, class_names):
        self.class_num = len(class_names)
        self.Cross_entropy = nn.CrossEntropyLoss()
    """
    def ring_loss(self, y_true,y_pred):
        pred=torch.sqrt(torch.sum(torch.square(y_pred),axis=1))
        error=torch.mean(torch.square(
            # Loss for Knowns having magnitude greater than knownsMinimumMag
            y_true[:,0]*(torch.maximum(knownsMinimumMag-pred,0.))
            # Add two losses
            +
            # Loss for unKnowns having magnitude greater than unknownsMaximumMag
            y_true[:,1]*pred
        ))
        return error
    """
    
    def __call__(self, output, target_batch):
        output_softmax = F.softmax(output, dim=1)
        loss = 0
        for i, target in enumerate(target_batch):
            if target == self.class_num - 1:
                # background class
                div = 1/self.class_num
                for index in range(self.class_num - 1):
                    loss -= torch.log(output_softmax[i][index])*div 
            else:
                # CELoss
                loss -= torch.log(output_softmax[i][target])
                # print(loss)

        loss_mean = torch.div(loss,len(target_batch))
        return loss_mean
    
    """
    def __call__(self, output, target_batch):
        output_softmax = F.softmax(output, dim=1)
        loss = 0
        for i, target in enumerate(target_batch):
            if target == self.class_num - 1:
                # background class
                div = 1/self.class_num
                for index in range(self.class_num - 1):
                    loss -= torch.log(output_softmax[i][index])*div 
            else:
                # CELoss
                loss -= torch.log(output_softmax[i][target])
                # print(loss)

        loss_mean = loss/len(target_batch)
        return loss_mean
    """