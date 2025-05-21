# %%
# %%
import os
import sys
import numpy as np
import torch
# from torch.utils.data import TensorDataset, DataLoader

# DL-Kit
sys.path.append('/Users/hgoldwyn/Research/projects/SR_CNN/dl-kit') #TODO adjust path
sys.path.append('/projects/ecrpstats/dl-kit') #TODO adjust path
from dlkit.log.log_util import (logging_set_up, logging_get_logger)
# from dlkit.nets.mlp import MLPModel
from dlkit.opt.train import train_epochs
from dlkit.nets.conv2d import Conv2dUpscaleModelInterpolate

## Load model module
sys.path.append('/Users/hgoldwyn/Research/projects/SR_CNN/paper_repo/auxilary_modules') #TODO adjust path
sys.path.append('/projects/ecrpstats/distributional_SRCNN/auxilary_modules')
import data_loading
import plotting

import matplotlib.pyplot as plt

train_or_load = 'train'
# %%
hr_data_size = 64
lr_data_size = 8

# %%
epochs = 500
batch_size = 32
##
region = 0
subregion = 'all'

log_file_base_path = os.path.basename(__file__)[:-3]
print(f"Running {log_file_base_path}")

save_path = f'{log_file_base_path}.pt'

# %%
## import data
xtrainHR, xtestHR, xtrainLR, xtestLR = data_loading.import_data(
    region,
    subregion,
    train_fraction=.75, 
    order='(subregion, time)'
    )

# set device
device = (
    "cuda"
    if torch.cuda.is_available()
    # else "mps"
    # if torch.backends.mps.is_available()
    else "cpu"
    )
print(f"Using device: {device}")

# set up logging
if train_or_load == 'train':
    logging_set_up(log_file_base_path)
    logger = logging_get_logger()
else:
    logger = None

# create data loader
dataloader_train = data_loading.create_dataloader(xtrainLR, xtrainHR, batch_size)

# create model
input_size  = int(np.prod(xtrainLR.shape[1:]))
output_size = int(np.prod(xtrainHR.shape[1:]))
#
net = Conv2dUpscaleModelInterpolate(
    input_channels=1,
    hidden_conv_layers_channels=[32, 32, 32, 32, 32, 1],
    hidden_conv_layers_kernels=[3, 3, 3, 3, 3, 3],
    # hidden_conv_layers_activation=nn.ReLU(),
    # hidden_conv_layers_kwargs={},
    upscale_layer_indices=[1, 2, 3],
    output_activation=None,
    use_dropout=False,
    scale_factor=[2, 2, 2],
    padding_mode='replicate',
)
net.to(device)

# create optimizer and loss function
optimizer = torch.optim.Adam(
    net.parameters(), 
    # lr=0.0001
    lr=0.01,
    )

# Custom learning rate scheduler
class ExponentialDecayWithFloor(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, decay_rate, floor, last_epoch=-1):
        self.decay_rate = decay_rate
        self.floor = floor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Compute new learning rate, ensuring it doesn't go below the floor
        return [
            max(base_lr * self.decay_rate ** (self.last_epoch + 1), self.floor)
            for base_lr in self.base_lrs
        ]
    
# Initialize the scheduler
scheduler = ExponentialDecayWithFloor(optimizer, decay_rate=0.95, floor=0.0001)

## Define loss
def mse_loss(outputs, targets):
    y_mean = torch.flatten(outputs, 1)  ## Returns shape (1, 56**2)
    y_data = torch.flatten(targets, 1)
    
    y_diff = y_data - y_mean
    yt_y = torch.unsqueeze(y_diff, 1) @ torch.unsqueeze(y_diff, 2)
    return torch.mean(yt_y)

## ~~~~~~~~~~~
if train_or_load == 'train':
    ## Train model
    train_log = train_epochs(
        epochs, net, dataloader_train, optimizer, mse_loss,
        device=device, logger=logger, 
        checkpoint_dir=log_file_base_path,
        checkpoint_epochs=100,
        lr_scheduler=scheduler
        )
elif train_or_load == 'load':
    ## Load model
    # Get last model checkpoint
    runs_dir = f"./{log_file_base_path}"
    list_of_run_dirs = os.listdir(runs_dir)
    last_model_dir = max([f'./{runs_dir}/{run_dir}' for run_dir in list_of_run_dirs], key=os.path.getctime)
    list_of_model_checks = os.listdir(last_model_dir)
    last_model_check = max([f'{last_model_dir}/{model_check}' for model_check in list_of_model_checks], key=os.path.getctime)

    loaded = torch.load(last_model_check)
    net.load_state_dict(loaded['model_state_dict'])


# %% [markdown]
# Plot results
net.eval()

# %%
dataloader_test = data_loading.create_dataloader(xtestLR, xtestHR, batch_size)

# %%
def prep_predictions_for_plot(dataloader_test):
    with torch.no_grad():
        
        batched_inputs, batched_pred_mean, batched_pred_stdd, batched_targets = [], [], [], []

        for batch_idx, data in enumerate(dataloader_test):
            inputs, targets = data
            batched_inputs.append(inputs) 
            batched_targets.append(targets)

            ## Return mean and std estimates
            inputs = inputs.to(device)
            (
                pred_mean#, 
                # pred_stdd 
                ) = net(inputs)
            batched_pred_mean.append(pred_mean)
            # batched_pred_stdd.append(pred_stdd)

    inputs = torch.cat(batched_inputs).reshape((-1, lr_data_size, lr_data_size))
    pred_mean = torch.cat(batched_pred_mean).reshape((-1, hr_data_size, hr_data_size))
    # pred_stdd = torch.cat(batched_pred_stdd).reshape((-1, hr_data_size, hr_data_size))
    targets = torch.cat(batched_targets).reshape((-1, hr_data_size, hr_data_size))

    return (
        inputs, 
        pred_mean, 
        # pred_stdd, 
        targets
        )


# %%
(
    inputs, 
    pred_mean, 
    # pred_stdd, 
    targets
    ) = prep_predictions_for_plot(dataloader_test)

## Move back to cpu for plotting
inputs = inputs.cpu()
pred_mean = pred_mean.cpu()
targets = targets.cpu()

## Save error fields for analysis
error_fields = pred_mean - targets
np.save(f'{log_file_base_path}_errFields', error_fields)

## Get training error fields for 2-stage model
(
    train_inputs, 
    train_pred_mean, 
    # pred_stdd, 
    train_targets
    ) = prep_predictions_for_plot(dataloader_train)
## Move back to cpu
train_inputs = train_inputs.cpu()
train_pred_mean = train_pred_mean.cpu()
train_targets = train_targets.cpu()
## Save error fields
error_fields = train_pred_mean - train_targets
np.save(f'{log_file_base_path}_TrainErrFields', error_fields)


# %%
plotter = plotting.plot_w_interpolator(hr_data_size=64)

# %%
plotter.look_at_prediction_vs_interp(0, pred_mean, inputs, targets, match_colorbars=True)
plt.savefig(f'{log_file_base_path}_img0.pdf', bbox_inches='tight')

# %%
plotter.look_at_prediction_vs_interp(1, pred_mean, inputs, targets, match_colorbars=True)
plt.savefig(f'{log_file_base_path}_img1.pdf', bbox_inches='tight')
# %%
plotter.look_at_prediction_vs_interp(2, pred_mean, inputs, targets, match_colorbars=True)
plt.savefig(f'{log_file_base_path}_img2.pdf', bbox_inches='tight')
# %%
plotter.look_at_prediction_vs_interp(3, pred_mean, inputs, targets, match_colorbars=True)
plt.savefig(f'{log_file_base_path}_img3.pdf', bbox_inches='tight')
# %% 
plotter.look_at_prediction_vs_interp(4, pred_mean, inputs, targets, match_colorbars=True)
plt.savefig(f'{log_file_base_path}_img4.pdf', bbox_inches='tight')

# %% [markdown]
# Quantify MAPE across images

# %%
pred_mapes = []
inte_mapes = []

for i in range(pred_mean.shape[0]):
    pred_mapes.append(
        plotting.mean_absolute_percentage_error(hr_data=targets[i], pred=pred_mean[i])
        )
    
    _interp = plotter.gen_interp(inputs[i].numpy())

    inte_mapes.append(
        plotting.mean_absolute_percentage_error(hr_data=targets[i], pred=_interp)
        )

# %%
pred_mapes = np.asarray(pred_mapes)
inte_mapes = np.asarray(inte_mapes)

## Mape plots
plt.figure(dpi=600, figsize=(2,2))
plt.hist(pred_mapes-inte_mapes, bins=50)
plt.vlines(0, *plt.gca().get_ylim(), color='k', zorder=1, ls='--')
plt.ylabel('number of images')
plt.xlabel(r"Pred MAPE $-$ Interp MAPE")
plt.savefig(f'{log_file_base_path}_mape.pdf', bbox_inches='tight')

## cut out outliers to 2 Stds
plt.figure(dpi=600, figsize=(2,2))
dat = pred_mapes-inte_mapes
mean = np.mean(dat)
stdd = np.std(dat)
dat = dat[dat < mean + 2*stdd]
dat = dat[dat > mean - 2*stdd]
plt.hist(dat, bins=50)
plt.vlines(0, *plt.gca().get_ylim(), color='k', zorder=1, ls='--')
plt.ylabel('number of images')
plt.xlabel(r"Pred MAPE $-$ Interp MAPE")
plt.savefig(f'{log_file_base_path}_mapeTo2std.pdf', bbox_inches='tight')

## cut out outliers to 1 Std
plt.figure(dpi=600, figsize=(2,2))
dat = dat[dat < mean + 1*stdd]
dat = dat[dat > mean - 1*stdd]
plt.hist(dat, bins=50)
plt.vlines(0, *plt.gca().get_ylim(), color='k', zorder=1, ls='--')
plt.ylabel('number of images')
plt.xlabel(r"Pred MAPE $-$ Interp MAPE")
plt.savefig(f'{log_file_base_path}_mapeTo1std.pdf', bbox_inches='tight')

## Calculate grad Mape
def grad_mag(img):
    grad = np.gradient(img)
    return (grad[0]**2 + grad[1]**2)**0.5

pred_gm_mapes = []
inte_gm_mapes = []

for i in range(pred_mean.shape[0]):
    target_grad_mag = grad_mag(targets[i])
    pred_grad_mag = grad_mag(pred_mean[i])
    pred_gm_mapes.append(
        plotting.mean_absolute_percentage_error(hr_data=target_grad_mag, pred=pred_grad_mag)
        )
    
    _interp = plotter.gen_interp(inputs[i].numpy()).reshape((hr_data_size, hr_data_size))
    interp_grad_mag = grad_mag(_interp)

    inte_gm_mapes.append(
        plotting.mean_absolute_percentage_error(hr_data=target_grad_mag, pred=interp_grad_mag)
        )

# %%
pred_gm_mapes = np.asarray(pred_gm_mapes)
inte_gm_mapes = np.asarray(inte_gm_mapes)

## Save mape and grad mape to file
np.save(f'{log_file_base_path}_mape', pred_mapes)
np.save(f'{log_file_base_path}_interp_mape', inte_mapes)
np.save(f'{log_file_base_path}_grad_mape', pred_gm_mapes)
np.save(f'{log_file_base_path}_interp_grad_mape', inte_gm_mapes)

## Plot training curve
import re

# Initialize an empty list to store the extracted values
epochs = []
loss_means = []
loss_std = []

test_log = f"{log_file_base_path}_info.log"
# Open and read the file line by line
with open(test_log, 'r') as file:
    for line in file:
        # Use regex to search for "loss_mean" followed by a number
        match = re.search(r"epoch\s+([\d]+)", line)
        if match:
            # Append the extracted number to the list
            epochs.append(float(match.group(1)))
        match = re.search(r"loss_mean\s+([\d.e+-]+)", line)
        if match:
            # Append the extracted number to the list
            loss_means.append(float(match.group(1)))
        match = re.search(r"std\s+([\d.e+-]+)", line)
        if match:
            # Append the extracted number to the list
            loss_std.append(float(match.group(1)))

# Convert the list to a numpy array
epochs = np.array(epochs)
loss_means = np.array(loss_means)
loss_std = np.array(loss_std)

plt.figure(figsize=(2,2), dpi=600)
plt.plot(epochs, loss_means)
ylim = plt.gca().get_ylim()
# plt.yscale('log')
plt.fill_between(epochs, loss_means-loss_std, loss_means+loss_std, alpha=.25)
plt.ylim(*ylim)
plt.ylabel('MSE loss')
plt.xlabel('epoch')
plt.savefig(f'{log_file_base_path}_training_curve.pdf', bbox_inches='tight')


## Save training curve
np.save(f'{log_file_base_path}_training_curve', np.array([epochs, loss_means, loss_std]))
