import torch

class BaseActor(torch.nn.Module):
    def __init__(self, args, size):
        super(BaseActor, self).__init__()
        self.args = args
        self.size = size

        self.bias = torch.nn.Linear(size, 1, bias=False)
        for p in self.bias.parameters():
            p.data.fill_(1.)

    def forward(self, feature):
        logits = self.bias.weight * feature
        return logits