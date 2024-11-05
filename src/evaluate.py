import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.measure import label, regionprops

from src.dice_score import dice_coeff, multiclass_dice_coeff
from src.utils import pixel_to_micrometer


@torch.inference_mode()
def evaluate(net, dataloader: DataLoader, device: torch.device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the specified set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Evaluation round', unit='batch', position=0, leave=False):
        images, true_masks = batch

        if images.dim() == 5:
            # Reshape images and masks to merge tile dimension with batch dimension
            batch_size, num_tiles, channels, height, width = images.shape
            images = images.view(batch_size * num_tiles, channels, height, width)
            batch_size, num_tiles, height, width = true_masks.shape
            true_masks = true_masks.view(batch_size * num_tiles, height, width)

        # move images and labels to correct device and type
        images = images.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.float32)

        # predict the mask (shape: B x C x H x W)
        mask_preds = net(images)

        if net.n_classes == 1:
            assert true_masks.min() >= 0 and true_masks.max() <= 1, 'True mask indices should be in [0, 1]'
            mask_preds = (F.sigmoid(mask_preds) > 0.5)
            # Add an extra dimension
            true_masks = true_masks.unsqueeze(1)
            # compute the Dice score
            dice_score += dice_coeff(mask_preds, true_masks, reduce_batch_first=False)
        else:
            assert true_masks.min() >= 0 and true_masks.max() < net.n_classes, 'True mask indices should be in [0, n_classes)'
            # convert to one-hot format
            mask_preds = F.one_hot(mask_preds.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2)
            # compute the Dice score, ignoring background
            dice_score += multiclass_dice_coeff(mask_preds[:, 1:], true_masks[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)

@torch.inference_mode()
def evaluate_all_metrics(net, dataloader: DataLoader, device: torch.device, scale: int):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    mask_cells = 0
    detected_cells = 0
    cell_sizes = []
    cell_sizes_label = []

    for batch in tqdm(dataloader, total=num_val_batches, desc='Evaluation round', unit='batch', position=0, leave=False):
        images, true_masks = batch

        if images.dim() == 5:
            batch_size, num_tiles, channels, height, width = images.shape
            images = images.view(batch_size * num_tiles, channels, height, width)
            batch_size, num_tiles, height, width = true_masks.shape
            true_masks = true_masks.view(batch_size * num_tiles, height, width)

        images = images.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.float32)

        mask_preds = net(images)

        if net.n_classes == 1:
            assert true_masks.min() >= 0 and true_masks.max() <= 1, 'True mask indices should be in [0, 1]'
            mask_preds = (F.sigmoid(mask_preds) > 0.5)
            true_masks = true_masks.unsqueeze(1)
            dice_score += dice_coeff(mask_preds, true_masks, reduce_batch_first=False)
        else:
            assert true_masks.min() >= 0 and true_masks.max() < net.n_classes, 'True mask indices should be in [0, n_classes)'
            mask_preds = F.one_hot(mask_preds.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2)
            dice_score += multiclass_dice_coeff(mask_preds[:, 1:], true_masks[:, 1:], reduce_batch_first=False)

        mask_preds = mask_preds.cpu().detach().numpy()
        true_masks = true_masks.cpu().numpy()

        for i in range(len(images)):
            mask_pred_int = mask_preds[i].astype(np.int32)
            true_mask_int = true_masks[i].astype(np.int32)
            labeled_pred, num_cells_pred = label(mask_pred_int, return_num=True)
            labeled_label, num_cells_label = label(true_mask_int, return_num=True)

            cell_sizes.extend([region.area for region in regionprops(labeled_pred)])
            cell_sizes_label.extend([region.area for region in regionprops(labeled_label)])

            mask_cells += num_cells_label
            detected_cells += num_cells_pred

    cell_sizes = pixel_to_micrometer(cell_sizes, scale)
    cell_sizes_label = pixel_to_micrometer(cell_sizes_label, scale)

    cell_detection_rate = detected_cells / mask_cells if mask_cells > 0 else 0
    avg_cell_size = np.mean(cell_sizes) if cell_sizes else 0
    std_cell_size = np.std(cell_sizes) if cell_sizes else 0
    avg_cell_size_label = np.mean(cell_sizes_label) if cell_sizes_label else 0
    std_cell_size_label = np.std(cell_sizes_label) if cell_sizes_label else 0

    dice_score = dice_score / max(num_val_batches, 1)

    metrics = {
        'dice_score': dice_score.item(),
        'detection_rate': cell_detection_rate,
        'avg_pred_cell_size': avg_cell_size,
        'std_pred_cell_size': std_cell_size,
        'avg_mask_cell_size': avg_cell_size_label,
        'std_mask_cell_size': std_cell_size_label
    }

    net.train()
    return metrics

@torch.inference_mode()
def predict(net, inputs: torch.Tensor) -> torch.Tensor:
    if len(inputs.shape) == 3:
        inputs = inputs.unsqueeze(dim=0)
    net.eval()
    out = net(inputs)
    mask_preds = (F.sigmoid(out) > 0.5)
    return mask_preds


def predict_image(net, img_path, out_path):
    size = 256 # The size of the input images to the network
    img = Image.open(img_path)
    if img.width < size:
        ratio = size / img.width
        img = img.resize((size, int(ratio * img.height)))
    if img.height < size:
        ratio = size / img.height
        img = img.resize((int(ratio * img.width), size))

    w, h = img.width, img.height
    img = np.array(img)

    # Split image into regions
    xs = [x for x in range(0, w-size, size)] + [w - size]
    ys = [y for y in range(0, h-size, size)] + [h - size]
    crops = [torch.from_numpy(img[y:y+size, x:x+size]) for y in ys for x in xs]

    # Create inputs for network
    inputs = torch.cat([torch.unsqueeze(crop, dim=0) for crop in crops], dim=0)
    inputs = inputs.permute((0, 3, 1, 2)).type(torch.float)
    inputs /= 255

    mask_preds = predict(net, inputs)

    # Create mask for whole image
    mask = np.empty((h, w), dtype=np.bool_)
    for i, crop_mask in enumerate(mask_preds):
        y, x = ys[i//len(xs)], xs[i%len(xs)]
        mask[y:y+size, x:x+size] = crop_mask.numpy()

    # Merge the mask and the original image
    merged = np.ones((h, w, 4))*255
    merged[:,:,:-1] = img
    merged[mask, :-1] = merged[mask, :-1]*0.6 + np.array([255, 0, 0])*0.4
    merged = merged.astype(np.uint8)

    out_img = Image.fromarray(merged)
    out_img.save(out_path)
    return mask

