import os
import json
import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch

def plot_results_tiles(loader, model, device, plot_path, name, config):
    for batch_idx, (data, labels) in enumerate(loader):
        data = data.to(device)

        # merge batch and tile dimensions
        batch_size, tile_num, channels, height, width = data.shape
        data = data.view(batch_size * tile_num, channels, height, width)

        predictions = model(data)

        # restore batch and tile dimensions
        data = data.view(batch_size, tile_num, channels, height, width)
        batch_size, tile_num, height, width = labels.shape
        predictions = predictions.view(batch_size, tile_num, 1, height, width)

        binary_predictions = (torch.nn.functional.sigmoid(predictions) > 0.5)
        
        binary_predictions = binary_predictions.cpu().detach().numpy()
        data = data.cpu().detach().numpy()
        predictions = predictions.cpu().detach().numpy()

        batch_size = data.shape[0]
        for i in range(batch_size):
            index = loader.batch_size * batch_idx + i

            bio_size = 80 * config['input_scale']
            mask_size = 80 * config['mask_scale']
            overlap_rate = config['overlap_rate']

            label = np.squeeze(labels[i])
            binary_prediction = np.squeeze(binary_predictions[i])
            label = merge_tiles(label, mask_size, overlap_rate, mask=True)
            binary_prediction = merge_tiles(binary_prediction, mask_size, overlap_rate, mask=True)
            d = merge_tiles(data[i,:,-1], bio_size, overlap_rate, mask=False)
            prediction = merge_tiles(predictions[i, :, 0], mask_size, overlap_rate, mask=False)

            title = f'{name} {index}'
            plot_pred_image(d, label, prediction, binary_prediction, plot_path, title)


def plot_results_image(loader, model, device, plot_path, name, config):
    for batch_idx, (data, labels) in enumerate(loader):
        data = data.to(device)

        predictions = model(data)

        binary_predictions = (torch.nn.functional.sigmoid(predictions) > 0.5)
        
        batch_size = data.shape[0]
        for i in range(batch_size):
            index = loader.batch_size * batch_idx + i

            label = np.squeeze(labels[i])
            binary_prediction = np.squeeze(binary_predictions[i])
            d = data[i,-1]
            prediction = predictions[i, 0]

            title = f'{name} {index}'
            plot_pred_image(d, label, prediction, binary_prediction, plot_path, title)


def plot_pred_image(bio, mask, prediction, binary_prediction, plot_path, title):
    plt.figure(figsize=(45, 15))

    bio = bio.squeeze().cpu().detach().numpy()
    mask = mask.squeeze().cpu().detach().numpy()
    prediction = prediction.squeeze().cpu().detach().numpy()
    binary_prediction = binary_prediction.squeeze().cpu().detach().numpy()

    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
    colored_mask[mask == 1] = [1, 0, 0, 1]
    colored_mask[mask == 0] = [0, 0, 0, 0]

    colored_prediction = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
    colored_prediction[binary_prediction == 1] = [0, 0, 1, 1]
    colored_prediction[binary_prediction == 0] = [0, 0, 0, 0]

    plt.subplot(1, 4, 1)
    bio = cv2.resize(bio, (mask.shape[0], mask.shape[1]))
    plt.imshow(bio, cmap='gray') # cmap='viridis'?
    plt.imshow(colored_mask, alpha=0.6)
    plt.title('Biosensor and mask')
    
    plt.subplot(1, 4, 2)
    plt.imshow(prediction, cmap='gray')
    plt.imshow(colored_prediction, alpha=0.6)
    plt.title('Prediction with the binary')

    intercection = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
    intercection[(mask == 1) & (binary_prediction == 1)] = [0, 1, 0, 1]

    plt.subplot(1, 4, 3)
    plt.imshow(bio, cmap='gray')
    plt.imshow(colored_mask)
    plt.imshow(colored_prediction)
    plt.imshow(intercection)
    plt.title('Label and Prediction overlap on the biosensor')

    plt.subplot(1, 4, 4)
    plt.imshow(colored_mask)
    plt.imshow(colored_prediction)
    plt.imshow(intercection)
    plt.title('Label and Prediction overlap')
    
    red_patch = mpatches.Patch(color=[1, 0, 0, 1], label='Mask')
    blue_patch = mpatches.Patch(color=[0, 0, 1, 1], label='Prediction')
    green_patch = mpatches.Patch(color=[0, 1, 0, 1], label='Overlap')

    plt.legend(handles=[red_patch, blue_patch, green_patch], loc='upper right', bbox_to_anchor=(1.2, 1))
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.95)
    plt.suptitle(title)

    path = os.path.join(plot_path, title)
    plt.savefig(path)
    plt.close()


def plot_loader(loader, plot_func, plot_path, name):
    for batch_idx, (bio, mask) in enumerate(loader):
        bio = bio.cpu().numpy()
        mask = mask.cpu().numpy()

        batch_size = bio.shape[0]
        for i in range(batch_size):
            index = batch_size * batch_idx + i
            title = f'{name} {index}'
            plot_func(bio[i], mask[i], plot_path, title)
        

def plot_image(bio, mask, plot_path, title):
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.imshow(bio[-1])

    plt.subplot(1, 2, 2)
    plt.imshow(mask)

    plt.suptitle(title)
    
    path = os.path.join(plot_path, title)
    plt.savefig(path)
    plt.close()


def plot_tiles(bio_tiles, mask_tiles, plot_path, title):
    n, ch, h, w = bio_tiles.shape
    ratio = int(np.sqrt(n))

    bio_vmin, bio_vmax = bio_tiles.min(), bio_tiles.max()

    fig, axes = plt.subplots(ratio, 2 * ratio +1 , figsize=(22, 10))
    for i in range(ratio):
        for j in range(ratio):
            axes[i, j].imshow(bio_tiles[j + i * ratio, -1], vmin=bio_vmin, vmax=bio_vmax)
            axes[i, j].axis('off')
    for i in range(ratio):
        axes[i, ratio].axis('off')

    for i in range(ratio):
        for j in range(ratio):
            axes[i, j + ratio + 1].imshow(mask_tiles[j + i * ratio])
            axes[i, j + ratio + 1].axis('off')

    plt.suptitle(title)

    path = os.path.join(plot_path, title)
    plt.savefig(path)
    plt.close()

def plot_merged_tiles(bio_tiles, mask_tiles, plot_path, title):
    bio = merge_tiles(bio_tiles[:,-1], 80 * config['input_scale'], config['overlap_rate'], mask=False)
    mask = merge_tiles(mask_tiles, 80 * config['mask_scale'], config['overlap_rate'], mask=True)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(bio)

    plt.subplot(1, 2, 2)
    plt.imshow(mask)

    plt.suptitle(title)

    path = os.path.join(plot_path, title)
    plt.savefig(path)
    plt.close()


def create_tiles(bio, mask, ratio, overlap_rate=0):
    ch, bio_h, bio_w = bio.shape
    mask_h, mask_w = mask.shape
    bio_size = bio_h // ratio
    mask_size = mask_h // ratio
    # print(bio_size, mask_size)

    bio_stride = bio_size - int(bio_size * overlap_rate)
    mask_stride = mask_size - int(mask_size * overlap_rate)
    # print(bio_stride, mask_stride)

    bio_tiles = bio.unfold(1, bio_size, bio_stride).unfold(2, bio_size, bio_stride)
    bio_tiles = bio_tiles.permute(1, 2, 0, 3, 4).reshape(-1, ch, bio_size, bio_size)
    mask_tiles = mask.unfold(0, mask_size, mask_stride).unfold(1, mask_size, mask_stride)
    mask_tiles = mask_tiles.permute(0, 1, 2, 3).reshape(-1, mask_size, mask_size)

    return bio_tiles, mask_tiles

def merge_tiles(tiles, original_size, overlap_rate=0, mask=False):
    num_tiles, tile_size, _ = tiles.shape
    stride = tile_size - int(tile_size * overlap_rate)
    ratio = original_size // tile_size

    merged = torch.zeros(original_size, original_size)
    contribution_map = torch.zeros(original_size, original_size)

    idx = 0
    for i in range(0, original_size - tile_size + 1, stride):
        for j in range(0, original_size - tile_size + 1, stride):
            merged[i:i + tile_size, j:j + tile_size] += tiles[idx]
            contribution_map[i:i + tile_size, j:j + tile_size] += 1
            idx += 1

    merged /= contribution_map
    if mask:
        merged[merged >= 0.5] = 1
        merged[merged < 0.5] = 0
    return merged

# Is tiling and overlap rate okay if the input and mask scale is different
def test_tiling(config):
    if config['tiling']:
        tiling_rate = config['tiling_ratio']
        overlap_rate = config['overlap_rate']
        mask_size = 80 * config['mask_scale']
        input_size = 80 * config['input_scale']
        bio = torch.zeros((config['biosensor_length'], input_size, input_size))
        mask = torch.zeros((mask_size, mask_size))
        bio_tiles, mask_tiles = create_tiles(bio, mask, tiling_rate, overlap_rate)
        print(f"Tiling test\nBio tiles: {bio_tiles.shape}, Mask tiles: {mask_tiles.shape}\n")
        assert bio_tiles.shape[0] == mask_tiles.shape[0], "Tile number mismatch in data and mask"

def pixel_to_micrometer(pixel_areas, scale_factor, pixel_size=25):
    """
    Convert pixel areas to real-world areas.

    :param pixel_areas: List of areas in pixels.
    :param scale_factor: Scaling factor applied to the images.
    :param pixel_size: Size of one pixel in micrometers (default is 25 µm).
    :return: List of areas in square micrometers.
    """
    real_areas = []
    for area in pixel_areas:
        real_area = area * (pixel_size / scale_factor) ** 2
        real_areas.append(real_area)
    return real_areas


def find_best_epoch(log):
    best_epoch = 0
    best_dice = 0
    
    for data in log:
        if data['dice_score'] > 1:
            continue
        if data['dice_score'] >= best_dice:
            best_dice = data['dice_score']
            best_epoch = data['epoch']

    return best_epoch

        # log is a list of dicts
        # epoch, dice_score, detection_rate, avg_pred_cell_size, std_pred_cell_size, avg_mask_cell_size, std_mask_cell_size
        # Calculate the difference in average cell size and standard deviation
        # avg_cell_size_diff = abs(data['avg_pred_cell_size'] - data['avg_mask_cell_size'])
        # std_cell_size_diff = abs(data['std_pred_cell_size'] - data['std_mask_cell_size'])
        
        # # These values mean the model is definitely not the best
        # if data['dice_score'] > 1:
        #     data['dice_score'] = 0
        # if data['detection_rate'] > 1:
        #     data['detection_rate'] = 0

        # Define a composite score
        # score = (
        #     10000 * data['dice_score'] +  # Higher dice score is better
        #     2000 * data['detection_rate'] -  # Higher detection rate is better
        #     avg_cell_size_diff -  # Smaller difference in average cell size is better
        #     std_cell_size_diff    # Smaller difference in standard deviation is better
        # )
        # print(f'Epoch: {data["epoch"]}, Score: {score}')


def load_config(run_path):
    with open(os.path.join(run_path, 'config.json'), 'r') as f:
        loaded_config = json.load(f)

    loaded_config['mask_type'] = bool
    # Ezeket még nem tudom, hogy kellene visszaalakítani, ha kell egyáltalán
    # loaded_config['model'] = loaded_config['model']
    # loaded_config['down_conv'] = loaded_config['down_conv'])
    # loaded_config['up_conv'] = loaded_config['up_conv'])
    return loaded_config
