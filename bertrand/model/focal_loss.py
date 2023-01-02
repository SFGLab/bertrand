import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    """
    Focal loss implementation as nn.Module
    """

    def __init__(
        self,
        gamma: float = 0,
        alpha: float = None,
        size_average: bool = True,
        no_agg: bool = False,
    ):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.alpha = torch.Tensor([alpha, 1 - alpha])
        self.size_average = size_average
        self.no_agg = no_agg

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes focal loss fo a batch of observations
        :param input: model predictions
        :param target: true class labels
        :return: loss for every observation if `no_agg` is True, otherwise
            average loss if `size_average` is True, else sum of losses
        """
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.no_agg:
            return loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
