from .attack import Attack
import torch
from torch.autograd import Variable


class IFGSM(Attack):
    def __init__(self, model, eps, alpha, iteration):
        super(IFGSM, self).__init__(model)
        self.eps = eps
        self.alpha = alpha
        self.iteration = iteration

    def forward(self, x):
        x_adv = Variable(x.data, requires_grad=True)
        output = self.model(x_adv)
        value, label = torch.max(output, 1)

        for i in range(self.iteration):
            x_adv = Variable(x_adv.data, requires_grad=True)
            output = self.model(x_adv)
            cost = -self.criterion(output, label)
            self.model.zero_grad()
            cost.backward()

            sign_grad = x_adv.grad.sign()
            x_adv = x_adv - self.alpha * sign_grad

            x_adv = torch.where(x_adv > x + self.eps, x + self.eps, x_adv)
            x_adv = torch.where(x_adv < x - self.eps, x - self.eps, x_adv)
            x_adv = torch.clamp(x_adv, self.x_val_min, self.x_val_max)
        return x_adv.detach()
