import torch
from torch import nn
import torch.nn.functional as F

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
  
  class JensenShannonDivergence(nn.Module):
    def __init__(self):
        super(JensenShannonDivergence, self).__init__()

    def __call__(self, p: torch.Tensor, q: torch.Tensor, log_proba=True) -> torch.Tensor:
        if log_proba:
            # Convert from log-space to probability-space
            p_prob = p.exp()
            q_prob = q.exp()
        else:
            # Inputs are already in probability space
            p_prob = p
            q_prob = q
        
        m_prob = 0.5 * (p_prob + q_prob)
        
        # Convert m back to log-space for KL divergence
        m_log = m_prob.log()
        
        if log_proba:
            # Use p and q directly since they are log-probs
            kl_p_m = F.kl_div(p, m_log, reduction='batchmean', log_target=True)
            kl_q_m = F.kl_div(q, m_log, reduction='batchmean', log_target=True)
        else:
            # Need to take log(p) and log(q) since they are probabilities
            kl_p_m = F.kl_div(p.log(), m_log, reduction='batchmean', log_target=True)
            kl_q_m = F.kl_div(q.log(), m_log, reduction='batchmean', log_target=True)

        return 0.5 * (kl_p_m + kl_q_m)


class NCC(nn.Module):
    def __init__(self, loss_weights=None):
        super(NCC, self).__init__()
        self.loss_weights = loss_weights if loss_weights is not None else {}

    def forward(self, x: torch.Tensor, y: torch.Tensor, peaks_prob_signal: torch.Tensor) -> torch.Tensor:
        """
        Computes the normalized cross-correlation coefficient for input signals.
        The input signal x can be of shape [batch_size, seq_length] or [seq_length].
        The input signal y can be of shape [batch_size, seq_length] or [seq_length].
        The peaks_prob_signal is passed as a separate tensor of shape [batch_size, seq_length].
        Returns a mean value of the cross-correlation coefficients.
        """

        # Reshape x and y if they are 1D (seq_length only) to [1, seq_length]
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Shape becomes [1, seq_length]

        if y.dim() == 1:
            y = y.unsqueeze(0)  # Shape becomes [1, seq_length]

        # Ensure x and y have the same shape
        assert x.shape == y.shape, "X and Y must be of the same shape"
        
        # Calculate weights per batch sample based on peaks_prob_signal
        if self.loss_weights:
            breaths_per_window = torch.sum(peaks_prob_signal, dim=1).long()  # Ensure it's an integer
            weight_per_window = [self.loss_weights.get(int(breath), 1.0) for breath in breaths_per_window]
            weight_per_window = torch.tensor(weight_per_window, device=x.device)
        else:
            weight_per_window = torch.ones(x.shape[0], device=x.device)

        # Compute the mean along the sequence dimension (dim=1)
        x_mean = torch.mean(x, dim=1, keepdim=True)
        y_mean = torch.mean(y, dim=1, keepdim=True)

        # Subtract the mean from the inputs
        x_prime = x - x_mean
        y_prime = y - y_mean

        # Compute the covariance
        covariance = torch.sum(x_prime * y_prime, dim=1)

        # Compute the standard deviations
        std_x = torch.sqrt(torch.sum(x_prime ** 2, dim=1))
        std_y = torch.sqrt(torch.sum(y_prime ** 2, dim=1))

        # Compute the denominator and avoid division by zero
        denominator = std_x * std_y
        r_xy = torch.where(denominator != 0, covariance / denominator, torch.zeros_like(covariance))
        
        # Element-wise multiply with weights
        r_xy = r_xy * weight_per_window

        # Return the mean of the cross-correlation coefficients
        return torch.mean(r_xy)


class WeightedKLDivLoss(nn.Module):
    def __init__(self, loss_weights=None, reduction='batchmean', log_target=True):
        super(WeightedKLDivLoss, self).__init__()
        self.loss_weights = loss_weights if loss_weights is not None else {}
        self.kl_div = nn.KLDivLoss(reduction='none', log_target=log_target)  # Use 'none' to apply custom reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor, peaks_prob_signal: torch.Tensor) -> torch.Tensor:
        """
        Computes weighted KL divergence loss.
        `input` and `target` should have shape [batch_size, *].
        `peaks_prob_signal` is used to calculate `breaths_per_window` and apply sample weights.
        """
        # Compute KL divergence for each element (no reduction)
        kl_loss = self.kl_div(input, target)  # Shape: [batch_size, *]
        
        # Compute mean KL loss per batch sample
        kl_loss_per_batch = kl_loss.sum(dim=1)  # Sum over the feature dimension (shape: [batch_size])

        # Calculate weights per batch sample
        breaths_per_window = torch.sum(peaks_prob_signal, dim=1).long()
        weight_per_window = [self.loss_weights.get(int(breath), 1.0) for breath in breaths_per_window]
        weight_per_window = torch.tensor(weight_per_window, device=input.device)

        # Apply the weights
        weighted_kl_loss = kl_loss_per_batch * weight_per_window

        # Return the mean weighted KL divergence loss
        return torch.mean(weighted_kl_loss)


class WeightedL1Loss(nn.Module):
    def __init__(self, loss_weights=None):
        super(WeightedL1Loss, self).__init__()
        self.loss_weights = loss_weights if loss_weights is not None else {}

    def forward(self, input: torch.Tensor, target: torch.Tensor, peaks_prob_signal: torch.Tensor) -> torch.Tensor:
        """
        Computes weighted L1 loss.
        `input` and `target` should have shape [batch_size, *].
        `peaks_prob_signal` is used to calculate `breaths_per_window` and apply sample weights.
        """
        # Compute L1 loss for each batch sample
        l1_loss = torch.abs(input - target).mean(dim=1)  # Shape: [batch_size]

        # Calculate weights per batch sample
        breaths_per_window = torch.sum(peaks_prob_signal, dim=1).long()
        weight_per_window = [self.loss_weights.get(int(breath), 1.0) for breath in breaths_per_window]
        weight_per_window = torch.tensor(weight_per_window, device=input.device)

        # Apply the weights
        weighted_l1_loss = l1_loss * weight_per_window

        # Return the mean weighted L1 loss across the batch
        return torch.mean(weighted_l1_loss)