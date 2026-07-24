import numpy as np
import torch
from torch import nn
from torch.autograd import Function, Variable
import torch.nn.functional as F
from einops import rearrange
import logging

class NearestEmbedFunc(Function):
    """
    Input:
    ------
    x - (batch_size, emb_dim, *)
        Last dimensions may be arbitrary
    emb - (emb_dim, num_emb)
    """
    @staticmethod
    def forward(ctx, input, emb):
        if input.size(1) != emb.size(0):
            raise RuntimeError('invalid argument: input.size(1) ({}) must be equal to emb.size(0) ({})'.
                               format(input.size(1), emb.size(0)))

        # save sizes for backward
        ctx.batch_size = input.size(0)
        ctx.num_latents = int(np.prod(np.array(input.size()[2:])))
        ctx.emb_dim = emb.size(0)
        ctx.num_emb = emb.size(1)
        ctx.input_type = type(input)
        ctx.dims = list(range(len(input.size())))

        # expand to be broadcast-able
        x_expanded = input.unsqueeze(-1)
        num_arbitrary_dims = len(ctx.dims) - 2
        if num_arbitrary_dims:
            emb_expanded = emb.view(
                emb.shape[0], *([1] * num_arbitrary_dims), emb.shape[1])
        else:
            emb_expanded = emb

        # find nearest neighbors
        dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        _, argmin = dist.min(-1)
        shifted_shape = [input.shape[0], *
                         list(input.shape[2:]), input.shape[1]]
        result = emb.t().index_select(0, argmin.view(-1)
                                      ).view(shifted_shape).permute(0, ctx.dims[-1], *ctx.dims[1:-1])

        ctx.save_for_backward(argmin)
        return result.contiguous(), argmin

    @staticmethod
    def backward(ctx, grad_output, argmin=None):
        grad_input = grad_emb = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        if ctx.needs_input_grad[1]:
            argmin, = ctx.saved_variables
            latent_indices = torch.arange(ctx.num_emb).type_as(argmin)
            idx_choices = (argmin.view(-1, 1) ==
                           latent_indices.view(1, -1)).type_as(grad_output.data)
            n_idx_choice = idx_choices.sum(0)
            n_idx_choice[n_idx_choice == 0] = 1
            idx_avg_choices = idx_choices / n_idx_choice
            grad_output = grad_output.permute(0, *ctx.dims[2:], 1).contiguous()
            grad_output = grad_output.view(
                ctx.batch_size * ctx.num_latents, ctx.emb_dim)
            grad_emb = torch.sum(grad_output.data.view(-1, ctx.emb_dim, 1) *
                                 idx_avg_choices.view(-1, 1, ctx.num_emb), 0)
        return grad_input, grad_emb, None, None


def nearest_embed(x, emb):
    return NearestEmbedFunc().apply(x, emb)


class NearestEmbed(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim):#k,features
        super(NearestEmbed, self).__init__()
        self.weight = nn.Parameter(torch.rand(embeddings_dim, num_embeddings))
        std_index=2
        logging.info(" std_index: "+str(std_index))
        # print("std_index: ",std_index)
        nn.init.normal_(self.weight, mean=0.0, std=std_index*(embeddings_dim**-0.5))
        # print('max:{:.11f},min:{:.11f}'.format(self.weight.max(),self.weight.min()))

    def forward(self, x, weight_sg=False):
        """Input:
        ---------
        x - (batch_size, emb_size, *)
        """
        return nearest_embed(x, self.weight.detach() if weight_sg else self.weight)

# adapted from https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py#L25
# that adapted from https://github.com/deepmind/sonnet

class SimVQ1D(nn.Module):
    def __init__(self, n_e, e_dim, beta=0.25, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=self.e_dim**-0.5)
        for p in self.embedding.parameters():
            p.requires_grad = False
        
        self.embedding_proj = nn.Linear(self.e_dim, self.e_dim)
    
        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape
        
    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        
        # z = rearrange(z, 'b c h -> b h c').contiguous()
        assert z.shape[-1] == self.e_dim
        
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        
        quant_codebook = self.embedding_proj(self.embedding.weight)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(quant_codebook**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(quant_codebook, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = F.embedding(min_encoding_indices, quant_codebook).view(z.shape)
        perplexity = None
        min_encodings = None

        # preserve gradients
        z_q = z + (z_q - z).detach()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten


        return z_q, min_encoding_indices
    

# class NearestEmbedEMA(nn.Module):
#     def __init__(self, n_emb, emb_dim, decay=0.99, eps=1e-5):
#         super(NearestEmbedEMA, self).__init__()
#         self.decay = decay
#         self.eps = eps
#         self.embeddings_dim = emb_dim
#         self.n_emb = n_emb
#         self.emb_dim = emb_dim
#         embed = torch.rand(emb_dim, n_emb)
#         self.register_buffer('weight', embed)
#         self.register_buffer('cluster_size', torch.zeros(n_emb))
#         self.register_buffer('embed_avg', embed.clone())

#     def forward(self, x):
#         """Input:
#         ---------
#         x - (batch_size, emb_size, *)
#         """

#         dims = list(range(len(x.size())))
#         x_expanded = x.unsqueeze(-1)
#         num_arbitrary_dims = len(dims) - 2
#         if num_arbitrary_dims:
#             emb_expanded = self.weight.view(
#                 self.emb_dim, *([1] * num_arbitrary_dims), self.n_emb)
#         else:
#             emb_expanded = self.weight

#         # find nearest neighbors
#         dist = torch.norm(x_expanded - emb_expanded, 2, 1)
#         _, argmin = dist.min(-1)
#         shifted_shape = [x.shape[0], *list(x.shape[2:]), x.shape[1]]
#         result = self.weight.t().index_select(
#             0, argmin.view(-1)).view(shifted_shape).permute(0, dims[-1], *dims[1:-1])

#         if self.training:
#             latent_indices = torch.arange(self.n_emb).type_as(argmin)
#             emb_onehot = (argmin.view(-1, 1) ==
#                           latent_indices.view(1, -1)).type_as(x.data)
#             n_idx_choice = emb_onehot.sum(0)
#             n_idx_choice[n_idx_choice == 0] = 1
#             flatten = x.permute(
#                 1, 0, *dims[-2:]).contiguous().view(x.shape[1], -1)

#             self.cluster_size.data.mul_(self.decay).add_(
#                 1 - self.decay, n_idx_choice
#             )
#             embed_sum = flatten @ emb_onehot
#             self.embed_avg.data.mul_(self.decay).add_(
#                 1 - self.decay, embed_sum)

#             n = self.cluster_size.sum()
#             cluster_size = (
#                 (self.cluster_size + self.eps) /
#                 (n + self.n_emb * self.eps) * n
#             )
#             embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
#             self.weight.data.copy_(embed_normalized)

#         return result, argmin
