import torch
import numpy as np


class Accuracy:
    def __init__(self, topk=5, remove=2, no_background=False):
        self.true = 0
        self.num_eval = 0

        self.topk = topk
        self.remove = remove
        self.no_background = no_background

    def reset(self):
        self.true = 0
        self.num_eval = 0

    def match(self, results, targets):
        #except background
        results = results.detach().cpu().numpy()
        targets = targets.cpu().numpy()
        if not self.no_background:
            cnt = 0
            for i, d in enumerate(targets):
                if int(d) == int(self.remove):
                    targets = np.delete(targets, i-cnt, axis=0)
                    results = np.delete(results, i-cnt, axis=0)
                    cnt+=1
        targets = torch.from_numpy(targets)
        results = torch.from_numpy(results)
        num_batch = targets.size(0)

        # get top5 indices
        _, results_ = results.topk(self.topk, 1)
        targets_ = targets.unsqueeze(1).expand_as(results_)

        self.true = self.true + (results_ == targets_).sum().item()
        self.num_eval += num_batch

        #print('....................................', self.true * 100. / self.num_eval)

    def get_result(self):
        return (self.true * 100. / self.num_eval)