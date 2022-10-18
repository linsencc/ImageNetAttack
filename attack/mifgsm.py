from .attack import Attack
import torch
from torch.autograd import Variable


class MIFGSM(Attack):
    def __init__(self, model, eps, alpha, iteration, decay_factor):
        super(MIFGSM, self).__init__(model)
        self.alpha = alpha
        self.decay_factor = decay_factor
        self.iteration = iteration
        self.eps = eps

    def forward(self, x):
        x_adv = Variable(x.data, requires_grad=True)
        output = self.model(x_adv)
        value, label = torch.max(output, 1)

        grad_data = 0
        for i in range(self.iteration):
            x_adv = Variable(x_adv.data, requires_grad=True)
            output = self.model(x_adv)
            cost = -self.criterion(output, label)

            self.model.zero_grad()
            cost.backward()
            grad_data = self.decay_factor * grad_data + x_adv.grad / torch.norm(x_adv.grad, p=1)
            x_adv = x_adv - self.alpha * grad_data.sign()

            x_adv = torch.where(x_adv > x + self.eps, x + self.eps, x_adv)
            x_adv = torch.where(x_adv < x - self.eps, x - self.eps, x_adv)
            x_adv = torch.clamp(x_adv, self.x_val_min, self.x_val_max)
        return x_adv.detach()

