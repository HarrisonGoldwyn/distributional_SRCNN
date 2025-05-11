import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from mpl_toolkits.axes_grid1 import make_axes_locatable

## Define performance metrics
def mean_absolute_percentage_error(hr_data, pred):

    if type(hr_data) is not np.ndarray:
        hr_data = hr_data.numpy()
    if type(pred) is not np.ndarray:
        pred = pred.numpy()

    hr_data = hr_data.ravel()
    pred = pred.ravel()
    
    mape = np.sum(
        np.abs(
            (hr_data - pred)/hr_data
            )
        ) / len(hr_data.ravel())

    return mape

def add_colorbar(mappable):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

## Interpolation of LR for comparison to fit
class plot_w_interpolator(object):

    def __init__(self, lr_data_size=8, hr_data_size=56):

        self.xL = np.arange(1, lr_data_size+1)
        self.xH = np.arange(1, hr_data_size+1)

        self.lr_data_size = lr_data_size
        self.hr_data_size = hr_data_size

        # xH / hr_data_size * lr_data_size
        interp_points = ((self.xH - 1/2) / hr_data_size) * lr_data_size

        xg, yg = np.meshgrid(interp_points, interp_points, indexing='ij')
        self.positions = np.vstack([xg.ravel(), yg.ravel()])


    def gen_interp(self, lr_test):
        ## Generate and plot interpolation
        interper = RegularGridInterpolator(
            (self.xL-.5, self.xL-.5,), lr_test, bounds_error=False, fill_value=None, method='cubic')
        interp = interper(self.positions.T).reshape((self.hr_data_size, self.hr_data_size, 1))
        return interp

    def look_at_prediction_vs_interp(
            self,
            img_idx, 
            predictions, 
            lr_test_data,
            hr_test_data, 
            match_colorbars=True,
            ):
        fig, axs = plt.subplots(1, 4, dpi=600, figsize=(8, 3))

        lr_ax = axs[0]
        in_ax = axs[1]
        pr_ax = axs[2]
        hr_ax = axs[3]

        ## Get image by index
        hr_test = hr_test_data[img_idx]
        lr_test = lr_test_data[img_idx]
        pred = predictions[img_idx]

        ## Plot data to get max for colorbars
        hr_im = hr_ax.imshow(hr_test)
        cmin = hr_test.min()
        cmax = hr_test.max()
        
        ## Plot prediction, interpolation
        interp = self.gen_interp(lr_test.numpy())
        if match_colorbars:
            pr_im = pr_ax.imshow(pred, vmin=cmin, vmax=cmax)
            in_im = in_ax.imshow(interp, vmin=cmin, vmax=cmax)
            lr_im = lr_ax.imshow(lr_test, vmin=cmin, vmax=cmax)
        else:
            pr_im = pr_ax.imshow(pred)
            in_im = in_ax.imshow(interp)
            lr_im = lr_ax.imshow(lr_test)


        add_colorbar(pr_im)
        add_colorbar(hr_im)
        add_colorbar(lr_im)
        add_colorbar(in_im)

        lr_ax.set_title('LR data')
        in_ax.set_title('Interp.')
        pr_ax.set_title('Pred.')
        hr_ax.set_title('HR data')

        ## Add MAPE to figure
        mape = mean_absolute_percentage_error(hr_test, pred)
        pr_ax.text(0, -.4, f"MAPE = {mape:.3f}", transform=pr_ax.transAxes)

        interp_mape = mean_absolute_percentage_error(hr_test, interp)
        in_ax.text(0, -.4, f"MAPE = {interp_mape:.3f}", transform=in_ax.transAxes)

        plt.tight_layout()
        # return interp



# from mpl_toolkits.axes_grid1 import AxesGrid

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    # plt.register_cmap(cmap=newcmap)

    return newcmap