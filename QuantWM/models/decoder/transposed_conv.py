import torch
import einops
import torch.nn as nn
import torch.nn.functional as F

def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

def horizontal_forward(network, x, input_shape=(-1,), output_shape=(-1,)):
    batch_with_horizon_shape = x.shape[: -len(input_shape)]
    if not batch_with_horizon_shape:
        batch_with_horizon_shape = (1,)
    x = x.reshape(-1, *input_shape)
    x = network(x)
    x = x.reshape(*batch_with_horizon_shape, *output_shape)
    return x

def create_normal_dist(
    x,
    std=None,
    mean_scale=1,
    init_std=0,
    min_std=0.1,
    activation=None,
    event_shape=None,
):
    if std == None:
        mean, std = torch.chunk(x, 2, -1)
        mean = mean / mean_scale
        if activation:
            mean = activation(mean)
        mean = mean_scale * mean
        std = F.softplus(std + init_std) + min_std
    else:
        mean = x
    dist = torch.distributions.Normal(mean, std)
    if event_shape:
        dist = torch.distributions.Independent(dist, event_shape)
    return dist
    

class TransposedConvDecoder(nn.Module):
    def __init__(self, observation_shape=(3, 224, 224), emb_dim=512, activation=nn.ReLU, depth=32, kernel_size=5, stride=3):
        super().__init__()

        activation = activation()
        self.observation_shape = observation_shape
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.emb_dim = emb_dim

        self.network = nn.Sequential(
            nn.Linear(
                emb_dim, self.depth * 32
            ),
            nn.Unflatten(1, (self.depth * 32, 1)),
            nn.Unflatten(2, (1,1)),
            nn.ConvTranspose2d(
                self.depth * 32,
                self.depth * 8,
                self.kernel_size,
                self.stride,
                padding=1
            ),
            activation,
            nn.ConvTranspose2d(
                self.depth * 8,
                self.depth * 4,
                self.kernel_size,
                self.stride,
                padding=1
            ),
            activation,
            nn.ConvTranspose2d(
                self.depth * 4,
                self.depth * 2,
                self.kernel_size,
                self.stride,
                padding=1
            ),
            activation,
            nn.ConvTranspose2d(
                self.depth * 2,
                self.depth * 1,
                self.kernel_size,
                self.stride,
                padding=1
            ),
            activation,
            nn.ConvTranspose2d(
                self.depth * 1,
                self.observation_shape[0],
                self.kernel_size,
                self.stride,
                padding=1
            ),
            nn.Upsample(size=(observation_shape[1], observation_shape[2]), mode='bilinear', align_corners=False)
        )
        self.network.apply(initialize_weights)

    def forward(self, posterior):
        x = horizontal_forward(
            self.network, posterior, input_shape=[self.emb_dim],output_shape=self.observation_shape
        )
        dist = create_normal_dist(x, std=1, event_shape=len(self.observation_shape))
        img = dist.mean.squeeze(2)
        img = einops.rearrange(img, "b t c h w -> (b t) c h w")
        return img, torch.zeros(1).to(posterior.device) # dummy placeholder