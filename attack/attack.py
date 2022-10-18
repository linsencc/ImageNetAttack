from torch.nn import CrossEntropyLoss


class Attack(object):
    r"""
    Base class for all attacks.

    .. note::
        It will changes the model's training mode to `test` by `.eval()`
    """
    def __init__(self, model):
        r"""
        Initializes internal attack state.

        Arguments:
            model (torch.nn.Module): model to attack.
        """
        self.model = model
        self.x_val_min = 0
        self.x_val_max = 1
        self.criterion = CrossEntropyLoss()

    def forward(self, *input):
        r"""
        It defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def __call__(self, *input, **kwargs):
        self.model.eval()
        images = self.forward(*input, **kwargs)
        return images