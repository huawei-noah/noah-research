import copy
import random
import torch
from scipy.stats import truncnorm
import numpy as np

def normalize_images(img):
    if img.dim() == 2: 
        max_val = torch.max(img)
        return img / max_val
    elif img.dim() == 5: 
        bs, T, _, H, W = img.shape
        normalized_img = torch.zeros_like(img)
        for b in range(bs):
            for t in range(T):
                single_img = img[b, t, 0]
                max_val = torch.max(single_img)
                normalized_img[b, t, 0] = single_img / max_val
        return normalized_img
    else:
        raise ValueError("Unsupported input shape. Expected either (h, w) or (bs, T, 1, h, w).")
    
def generate_wall_layouts(wall_config):
    """
    Generate possible layouts for train and validation
    """
    img_size = wall_config.img_size
    wall_padding = wall_config.wall_padding
    door_padding = wall_config.door_padding

    fix_wall = wall_config.fix_wall
    fix_door_location = wall_config.fix_door_location
    fix_wall_location = wall_config.fix_wall_location
    num_train_layouts = wall_config.num_train_layouts

    def extract_min_max(code):
        vals = [int(x) for x in code.split("-") if x]
        if len(vals) == 2:
            min_val, max_val = vals[0], vals[1]
        elif len(vals) == 1:
            min_val, max_val = vals[0], vals[0]
        else:
            return []
        return list(range(min_val, max_val + 1))

    exclude_wall_train = extract_min_max(wall_config.exclude_wall_train)
    exclude_door_train = extract_min_max(wall_config.exclude_door_train)
    only_wall_val = extract_min_max(wall_config.only_wall_val)
    only_door_val = extract_min_max(wall_config.only_door_val)

    "Arrays must all be empty or all be non-empty"
    assert all(
        len(arr) == 0
        for arr in [
            exclude_wall_train,
            exclude_door_train,
            only_wall_val,
            only_door_val,
        ]
    ) or all(
        len(arr) > 0
        for arr in [
            exclude_wall_train,
            exclude_door_train,
            only_wall_val,
            only_door_val,
        ]
    )

    if fix_wall:
        layouts = {
            f"v_wall{fix_wall_location}_door{fix_door_location}": {
                "type": "v",
                "wall_pos": fix_wall_location,
                "door_pos": fix_door_location,
            }
        }
        return layouts, None

    wall_x_values = list(range(wall_padding, img_size - wall_padding))
    door_y_values = list(range(door_padding, img_size - door_padding))

    layouts = {}
    other_layouts = {}

    for wall_pos in wall_x_values:
        for door_pos in door_y_values:
            code = f"wall{wall_pos}_door{door_pos}"

            to_exclude_for_train = (
                wall_pos in exclude_wall_train and door_pos in exclude_door_train
            )
            to_include_for_val = wall_pos in only_wall_val and door_pos in only_door_val

            if not to_exclude_for_train:
                layouts[f"v_{code}"] = {
                    "type": "v",
                    "wall_pos": wall_pos,
                    "door_pos": door_pos,
                }

                layouts[f"h_{code}"] = {
                    "type": "h",
                    "wall_pos": wall_pos,
                    "door_pos": door_pos,
                }

            if to_include_for_val:
                other_layouts[f"v_{code}"] = {
                    "type": "v",
                    "wall_pos": wall_pos,
                    "door_pos": door_pos,
                }

                other_layouts[f"h_{code}"] = {
                    "type": "h",
                    "wall_pos": wall_pos,
                    "door_pos": door_pos,
                }

    if not wall_config.train and exclude_wall_train:
        return other_layouts, None
    else:
        return layouts, None


def sample_uniformly_between(a, b):
    """
    Generate tensor c of the same shape as a and b, where each element of c
    is sampled uniformly between a[i] and b[i].

    Args:
    - a (torch.Tensor): Input tensor of shape (bs).
    - b (torch.Tensor): Input tensor of shape (bs), where each element is guaranteed to be bigger than a[i].

    Returns:
    - c (torch.Tensor): Output tensor of the same shape as a and b, with elements sampled uniformly.
    """
    # Generate random numbers from uniform distribution between a[i] and b[i]
    random_values = torch.FloatTensor(a.size()).uniform_(0, 1).to(a.device)

    # Scale the random values to fit the range [a[i], b[i]]
    c = a + (b - a) * random_values

    return c


def sample_truncated_norm(upper_bound, lower_bound, mean, std=1.4):
    """
    Generate tensor of the same shape as mean, where each element is
    sampled from a truncated norm distribution defined by upper_bound[i],
    lower_bound[i], and mean mean[i]
    """
    # Convert PyTorch tensors to NumPy
    upper_bound_np = upper_bound.cpu().float().numpy()
    lower_bound_np = lower_bound.cpu().float().numpy()
    mean_np = mean.cpu().float().numpy()

    # Initialize the output array
    samples = np.zeros_like(mean_np)

    # Sample from the truncated normal distribution
    for i in range(len(mean_np)):
        # Calculate the bounds in terms of standard deviations
        a, b = (lower_bound_np[i] - mean_np[i]) / std, (
            upper_bound_np[i] - mean_np[i]
        ) / std
        # Sample from truncnorm
        samples[i] = truncnorm.rvs(a, b, loc=mean_np[i], scale=std)

    # Convert back to PyTorch tensor
    return torch.from_numpy(samples)
