import torch
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
import imageio
import cfg


def print_log(message):
    print(message, flush=True)
    if cfg.logger:
        cfg.logger.write(str(message) + '\n')


# Plot losses
def plot_loss(d_losses, g_losses, num_epochs, save=True, save_dir='results/', show=False):
    fig, ax = plt.subplots()
    ax_ = ax.twinx()
    ax.set_xlim(0, num_epochs)
    # ax.set_ylim(0, max(np.max(g_losses), np.max(d_losses)) * 1.1)
    plt.xlabel('# of Epochs')
    ax.set_ylabel('Generator loss values')
    ax_.set_ylabel('Discriminator loss values')
    ax.plot(g_losses, label='Generator')
    ax_.plot(d_losses, label='Discriminator')
    plt.legend()

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + 'Loss_values_epoch_{:d}'.format(num_epochs) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


def plot_test_result_single(image, image_idx, save_dir='results/', fig_size=(5, 5)):
    # fig_size = (target.size(2) / 100, target.size(3) / 100)
    # fig, ax = plt.subplots(figsize=fig_size)
    fig, ax = plt.subplots()
    img = image

    ax.axis('off')
    ax.set_adjustable('box')
    # Scale to 0-255
    img = (((img[0] - img[0].min()) * 255) / (img[0].max() - img[0].min()))\
        .numpy().transpose(1, 2, 0).astype(np.uint8)
    ax.imshow(img, cmap=None, aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)

    save_path = save_dir
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, '%03d.png' % image_idx)
    fig.subplots_adjust(bottom=0)
    fig.subplots_adjust(top=1)
    fig.subplots_adjust(right=1)
    fig.subplots_adjust(left=0)

    foo_fig = plt.gcf()
    foo_fig.set_size_inches(5, 5)
    foo_fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.savefig(save_path)
    plt.close()


def plot_test_result(input, target, gen_image, image_idx, img_title, epoch, training=True,
                     save=True, save_dir='results/', show=False, fig_size=(5, 5)):
    if input is not None:
        fig_size = (target.size(2) * 3 / 100, target.size(3) / 100)
        imgs = [input, gen_image, target]
        fig, axes = plt.subplots(1, 3, figsize=fig_size)
    else:
        fig_size = (target.size(2) * 2 / 100, target.size(3) / 100)
        imgs = [gen_image, target]
        fig, axes = plt.subplots(1, 2, figsize=fig_size)

    for ax, img in zip(axes.flatten(), imgs):
        ax.axis('off')
        ax.set_adjustable('box')
        # Scale to 0-255
        img = (((img[0] - img[0].min()) * 255) / (img[0].max() - img[0].min()))\
            .numpy().transpose(1, 2, 0).astype(np.uint8)
        ax.imshow(img, cmap=None, aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)

    # save figure
    if save:
        # save_path = os.path.join(save_dir, str(image_idx))
        save_path = save_dir
        os.makedirs(save_path, exist_ok=True)
        if training:
            save_path = os.path.join(save_path, '%03d_%s.png' % (image_idx, img_title))
        else:
            save_path = os.path.join(save_path, 'Test_%03d_%s.png' % (image_idx, img_title))
        fig.subplots_adjust(bottom=0)
        fig.subplots_adjust(top=1)
        fig.subplots_adjust(right=1)
        fig.subplots_adjust(left=0)
        plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close()


def maybe_download(model_name, model_url, model_dir=None, map_location=None):
    import os, sys
    from six.moves import urllib
    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
        model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = '{}.pth.tar'.format(model_name)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        url = model_url
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urllib.request.urlretrieve(url, cached_file)
    if '152' in cached_file:
        cached_file = '/home/anbang/.cache/torch/checkpoints/resnet152-b121ed2d.pth'
    return torch.load(cached_file, map_location=map_location)


# Make gif
def make_gif(dataset, num_epochs, save_dir='results/'):
    gen_image_plots = []
    for epoch in range(num_epochs):
        # plot for generating gif
        save_fn = save_dir + 'Result_epoch_{:d}'.format(epoch + 1) + '.png'
        gen_image_plots.append(imageio.imread(save_fn))

    imageio.mimsave(save_dir + dataset + '_pix2pix_epochs_{:d}'.format(num_epochs) \
        + '.gif', gen_image_plots, fps=5)


def init_lists(length):
    lists = []
    for l in range(length):
        lists.append([])
    return lists


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
