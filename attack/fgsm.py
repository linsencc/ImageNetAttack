from .attack import Attack
import torch
from torch.autograd import Variable


class FGSM(Attack):
    def __init__(self, model, eps):
        super(FGSM, self).__init__(model)
        self.eps = eps

    def forward(self, x):
        x = Variable(x.data, requires_grad=True)
        output = self.model(x)
        value, predict = torch.max(output, 1)

        cost = -self.criterion(output, predict)
        self.model.zero_grad()
        cost.backward()

        sign_grad = x.grad.sign()
        x_adv = x - self.eps * sign_grad
        x_adv = torch.clamp(x_adv, min=-1, max=1)
        return x_adv.detach()
