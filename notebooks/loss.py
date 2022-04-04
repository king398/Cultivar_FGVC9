from torch import nn
import torch


class categorical_focal_loss_with_label_smoothing(nn.Module):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha*((1-p)^gamma)*log(p)
        y_ls = (1 - α) * y_hot + α / classes
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
        ls    -- label smoothing parameter(alpha)
        classes     -- No. of classes
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
        ls    -- 0.1
        classes     -- 4
    """

    def __init__(self, alpha, gamma, ls, classes):
        super(categorical_focal_loss_with_label_smoothing, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = torch.tensor(gamma)
        self.ls = ls
        self.classes = classes

    def forward(self, y_true, y_pred):
        # Define epsilon so that the backpropagation will not result in NaN
        # for 0 divisor case
        epsilon = torch.tensor(1e-07)
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # label smoothing
        # y_pred = torch.softmax(y_pred, dim=1)
        y_pred_ls = (1 - self.ls) * y_pred + self.ls / self.classes
        # Clip the prediction value
        y_pred_ls = torch.clamp(y_pred_ls, epsilon, 1.0 - epsilon)
        # Calculate cross entropy
        cross_entropy = -y_true * torch.log(y_pred_ls)
        # Calculate weight that consists of  modulating factor and weighting factor
        weight = self.alpha * y_true * torch.pow((1 - y_pred_ls), self.gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = torch.sum(loss, dim=1)
        return loss
