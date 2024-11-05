import os
import gc
import json

import torch
from torch.utils.data import DataLoader
from torchinfo import summary

from src.utils import find_best_epoch, plot_results_image, plot_results_tiles
from src.dataset import create_folds
from src.train import train_model
from src.evaluate import evaluate_all_metrics
from src.models import UNet, UNet2, UNet4, UNet8, UNet2Shallow, UNet4Shallow, UNet8Shallow
from src.model_parts import SingleConv, DoubleConv, TripleConv


def run_config(config):
    root_path = os.path.join(config['save_path'], config['run_name'])
    os.makedirs(root_path, exist_ok=True)

    save_config = config.copy()
    save_config['mask_type'] = config['mask_type'].__name__
    save_config['model'] = config['model'].__name__
    save_config['down_conv'] = config['down_conv'].__name__
    save_config['up_conv'] = config['up_conv'].__name__
    json.dump(save_config, open(os.path.join(root_path, 'config.json'), 'w'), indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    torch.manual_seed(42)

    folds, test_dataset = create_folds(config)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    print("Datasets created!")

    for i in range(config['k']):
        print(f'Starting run {i+1}...')
        run_path = os.path.join(root_path, f'run_{i+1}')
        os.makedirs(run_path, exist_ok=True)

        plot_path = os.path.join(run_path, 'plots')
        os.makedirs(plot_path, exist_ok=True)
        model_checkpoint_path = os.path.join(run_path, 'model_checkpoints')
        os.makedirs(model_checkpoint_path, exist_ok=True)

        mean, std, train_dataset, val_dataset = folds[i]
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

        model = config['model'](n_channels=config['biosensor_length'], n_classes=1, 
                                down_conv=config['down_conv'], up_conv=config['up_conv'], 
                                mean=mean, std=std, bilinear=config['bilinear'])
        print(model.__class__.__name__)

        if i == 0:
            # Save model summary only once
            model_summary = summary(model, depth=5)
            with open(os.path.join(root_path, 'model_summary.txt'), 'w') as f:
                f.write(str(model_summary))

        model = model.to(device)

        try:
            log = train_model(
                model,
                device,
                train_loader,
                val_loader,
                config,
                amp=True,
                checkpoint_dir=model_checkpoint_path,
                wandb_dir=config['save_path'],
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print('Detected OutOfMemoryError!')

        # Save training log
        json.dump(log, open(os.path.join(run_path, 'log.json'), 'w'), indent=4)
        print()
        best_epoch = find_best_epoch(log)
        print(f'\nBest epoch: {best_epoch}')
        best_checkpoint = torch.load(os.path.join(model_checkpoint_path, f'checkpoint_epoch{best_epoch}.pth'))
        lr = best_checkpoint.pop('learning_rate')
        model.load_state_dict(best_checkpoint)
        model = model.to(device)

        best_model_path = os.path.join(run_path, 'best_model.pth')
        torch.jit.script(model).save(best_model_path)

        test_metrics = evaluate_all_metrics(model, test_loader, device, config['mask_scale'])
        print('Test metrics:', json.dumps(test_metrics, indent=4))
        json.dump(test_metrics, open(os.path.join(root_path, f'test_result_run{i+1}.json'), 'w'), indent=4)

        # plot test results
        if config['tiling']:
            plot_results_tiles(test_loader, model, device, plot_path, 'Test image', config)
        else:
            plot_results_image(test_loader, model, device, plot_path, 'Test image', config)

        del train_loader, val_loader
        del train_dataset, val_dataset
        del model
        torch.cuda.empty_cache()
        gc.collect()

        print(f'Run {i+1} finished!')

    del test_loader, test_dataset
    torch.cuda.empty_cache()
    gc.collect()

    print('All runs finished!')