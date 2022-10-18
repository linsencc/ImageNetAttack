import torch
from .attack import Attack


class DeepFool(Attack):
    """Reproduce DeepFool
    in the paper 'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks'
    """
    def __init__(self, model, eps, iteration):
        super(DeepFool, self).__init__(model)
        self.iteration = iteration
        self.eps = eps

    def forward(self, img):
        ori_img = torch.clone(img)
        img.requires_grad = True
        output = self.model(img)[0]
        num_classes = len(output)

        _, first_predict = torch.max(output, 0)
        first_max = output[first_predict]
        grad_first = torch.autograd.grad(first_max, img)[0]

        for _ in range(self.iteration):
            img.requires_grad = True
            output = self.model(img)[0]
            _, predict = torch.max(output, 0)

            if predict != first_predict:
                img = torch.clamp(img, min=0, max=1).detach()
                break

            r = None
            min_value = None

            for k in range(num_classes):
                if k == first_predict:
                    continue

                k_max = output[k]
                grad_k = torch.autograd.grad(k_max, img, retain_graph=True, create_graph=True)[0]

                prime_max = k_max - first_max
                grad_prime = grad_k - grad_first
                value = torch.abs(prime_max) / torch.norm(grad_prime)

                if r is None:
                    r = (torch.abs(prime_max) / (torch.norm(grad_prime) ** 2)) * grad_prime
                    min_value = value

                if min_value > value:
                    r = (torch.abs(prime_max) / (torch.norm(grad_prime) ** 2)) * grad_prime
                    min_value = value

            img = img + r
            img = torch.where(img > ori_img + self.eps, ori_img + self.eps, img + r)
            img = torch.where(img < ori_img - self.eps, ori_img - self.eps, img + r)
            img = torch.clamp(img, min=0, max=1).detach()

        return img.detach()
