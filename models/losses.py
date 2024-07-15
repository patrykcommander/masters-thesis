import torch

class WeightedBCELoss(torch.nn.Module):
  def __init__(self, negative_weight=None, positive_weight=None):
    super(WeightedBCELoss, self).__init__()
    self.w_n = negative_weight
    self.w_p = positive_weight

  def forward(self, prediction, ground_truth, epsilon=1e-7):
    if self.w_n is None or self.w_p is None:
        # Calculate weights for the current batch
        pos_weight = (ground_truth == 1).float().sum()
        neg_weight = (ground_truth == 0).float().sum()
        total = pos_weight + neg_weight
        self.w_p = neg_weight / total
        self.w_n = pos_weight / total
    
    loss_pos = -1 * self.w_p * (ground_truth * torch.log(prediction + epsilon)).mean()
    loss_neg = -1 * self.w_n * ((1 - ground_truth) * torch.log(1 - prediction + epsilon)).mean()
  
    return loss_pos + loss_neg