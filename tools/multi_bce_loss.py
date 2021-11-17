import torch
import torch.nn as nn
class multi_bce_loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss=nn.BCEWithLogitsLoss()
    def forward(self,dp4,dp3,dp2,dp1,label):
        loss_dp4=self.loss(dp4,label)
        loss_dp3=self.loss(dp3,label)
        loss_dp2=self.loss(dp2,label)
        loss_dp1=self.loss(dp1,label)
        loss=0.4*loss_dp4+0.3*loss_dp3+0.2*loss_dp2+0.1*loss_dp1
        return loss