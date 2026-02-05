import numpy as np
import h5py


def store_data(filename, data, action):
    """
    action: (action_dim,)
    imgs_list: (T, num_cameras, H, W, 5)
    particle_pos_list: (T, N, 3)
    eef_states_list: (T, 14)
    """
    # load data
    imgs_list, particle_pos_list, eef_states_list = data
    imgs_list_np, particle_pos_list_np, eef_states_list_np = (
        np.array(imgs_list),
        np.array(particle_pos_list),
        np.array(eef_states_list),
    )

    # stat
    T, n_cam = imgs_list_np.shape[:2]
    n_particles = particle_pos_list_np.shape[1]

    # process images
    color_imgs, depth_imgs = process_imgs(imgs_list_np)
    
    # init episode data
    episode = {
        "info": {"n_cams": n_cam, "timestamp": T, "n_particles": n_particles},
        "action": action,
        "positions": particle_pos_list_np,
        "eef_states": eef_states_list_np,
        "observations": {"color": color_imgs, "depth": depth_imgs},
    }

    # save to h5py
    save_data(filename, episode)


def process_imgs(imgs_list):
    T, n_cam, H, W, _ = imgs_list.shape
    color_imgs = {}
    depth_imgs = {}

    for cam_idx in range(n_cam):
        img = imgs_list[:, cam_idx]  # (T, H, W, 5)
        color_imgs[f"cam_{cam_idx}"] = img[:, :, :, :3][..., ::-1]  # (T, H, W, 3)
        depth_imgs[f"cam_{cam_idx}"] = (img[:, :, :, -1] * 1000).astype(
            np.uint16
        )  # (T, H, W)

    assert color_imgs["cam_0"].shape == (T, H, W, 3)
    assert depth_imgs["cam_0"].shape == (T, H, W)

    return color_imgs, depth_imgs


def save_data(filename, save_data):
    with h5py.File(filename, "w") as f:
        for key, value in save_data.items():
            if key in ["observations"]:
                for sub_key, sub_value in value.items():
                    for subsub_key, subsub_value in sub_value.items():
                        f.create_dataset(
                            f"{key}/{sub_key}/{subsub_key}", data=subsub_value
                        )
            elif key in ["info"]:
                for sub_key, sub_value in value.items():
                    f.create_dataset(f"{key}/{sub_key}", data=sub_value)
            else:
                f.create_dataset(key, data=value)


def load_data(filename):
    data = {}
    with h5py.File(filename, "r") as f:
        for key in f.keys():
            if key in ["observations"]:
                data[key] = {}
                for sub_key in f[key].keys():
                    data[key][sub_key] = {}
                    for subsub_key in f[key][sub_key].keys():
                        data[key][sub_key][subsub_key] = f[key][sub_key][subsub_key][()]
            elif key in ["info"]:
                data[key] = {}
                for sub_key in f[key].keys():
                    data[key][sub_key] = f[key][sub_key][()]
            else:
                data[key] = f[key][()]
    return data
