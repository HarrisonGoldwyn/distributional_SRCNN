"""
Stage 2 (rewritten): Fit image-specific *correlation-shape* Fourier parameters
and then re-inject image-specific variance so C(r=0) matches the residual energy.

Key fixes vs your previous Stage 2:
  1) Per-image de-mean residuals (removes DC/mean drift).
  2) Per-image variance normalization: fit correlation shape only.
  3) Prior acts on *shape only* (normalized spectrum), so it cannot pin C(0).
  4) Explicit kappa weight (no hidden "/0.5" in the std).
  5) Clean separation: save (a) shape params, (b) per-image variance.

Notes:
  - This keeps your explicit basis-matrix approach to minimize disruption.
  - If you later want the “fully correct + fast” version, we can rewrite in FFT-space.
"""

import os
import sys
import numpy as np
import torch

# ---- paths (yours) ----
sys.path.append('/Users/hgoldwyn/Research/projects/SR_CNN/dl-kit')
sys.path.append('/projects/ecrpstats/dl-kit')

sys.path.append('/Users/hgoldwyn/Research/projects/SR_CNN/paper_repo/auxilary_modules')
sys.path.append('/projects/ecrpstats/distributional_SRCNN/auxilary_modules')
import data_loading

# ------------------------
# Config
# ------------------------
hr_data_size = 64
N = hr_data_size
min_k = 0
max_k = N // 2 + 1  # 0..32 inclusive -> 33x33 modes
epochs = 150
lr = 0.1
kappa = 0.5                 # explicit weight on prior
eps = 1e-12                 # numerical stability
out_dir = "output"
os.makedirs(out_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# ------------------------
# Load data (train residual fields)
# ------------------------
region = 0
subregion = "all"

xtrainHR, xtestHR, xtrainLR, xtestLR = data_loading.import_data(
    region, subregion, train_fraction=.75, order='(subregion, time)'
)

train_mse_error_fields = np.load(
    "/projects/ecrpstats/distributional_SRCNN/sr_ordered_train0p75/"
    "stage_1/output_data/mse_5l_i123_c32s_padR_schLrG0p95_reg0_TrainErrFields.npy"
)

# Shape: [n_img, 64, 64]
errs = train_mse_error_fields.astype(np.float32)
n_img = errs.shape[0]
assert errs.shape[1:] == (N, N)

# ------------------------
# Per-image mean removal and variance extraction (Option A)
# ------------------------
# IMPORTANT: this var is what you want C(0) to match (up to your FFT normalization)
img_mean = errs.mean(axis=(1, 2), keepdims=True)
errs_demean = errs - img_mean

img_var = (errs_demean ** 2).mean(axis=(1, 2))  # shape [n_img]
img_std = np.sqrt(img_var + 1e-30)

# Fit correlation-shape only: normalize each image to unit variance
errs_norm = errs_demean / img_std[:, None, None]

print("Residual variance stats (demeaned):",
      img_var.min(), img_var.mean(), img_var.max())

# Move to torch
err_fields_tensor = torch.tensor(errs_norm, dtype=torch.float32, device=device)
zeros_tensor = torch.zeros((1, N, N), dtype=torch.float32, device=device)

# ------------------------
# Build complex DFT basis (same as you had, but explicitly)
# ------------------------
def compl_dft_basis(mat_x, mat_y, kx, ky, N):
    return np.exp(1j * 2*np.pi * (kx*mat_x + ky*mat_y) / N)

x = np.arange(N)
xg = np.tile(x, N)
yg = np.repeat(x, N, axis=0)
mat_xg = xg.reshape((N, N))
mat_yg = yg.reshape((N, N))

basis_function_k_idx = []
basis_functions = []

for kx in range(min_k, max_k):
    for ky in range(min_k, max_k):
        basis_functions.append(compl_dft_basis(mat_xg, mat_yg, kx, ky, N))
        basis_function_k_idx.append((kx, ky))

# basis_functions: [Nmodes, N, N] -> reshape to [Npix, Nmodes]
basis_functions = np.asarray(basis_functions)               # [Nmodes, N, N]
basis_functions = basis_functions.reshape((-1, N*N)).T      # [Npix, Nmodes]
Nmodes = basis_functions.shape[1]
print("Nmodes:", Nmodes)

basis_functions_tensor = torch.tensor(
    basis_functions, dtype=torch.complex64, device=device
)

# Normalized B (keep consistent with your earlier code)
B = basis_functions_tensor / N
B_conjT = torch.conj(basis_functions_tensor.T) / N

# ------------------------
# Load global prior params + std (as before)
# ------------------------
global_fit_params = np.load(
    "/projects/ecrpstats/distributional_SRCNN/sr_ordered_train0p75/"
    "stage_2/output_2a/anal_sln_global_params.npy"
).astype(np.float32)

global_fit_param_stdd = np.load(
    "/projects/ecrpstats/distributional_SRCNN/sr_ordered_train0p75/"
    "stage_2/output_2a/img_spec_params_std.npy"
).astype(np.float32)

global_fit_params_t = torch.tensor(global_fit_params, dtype=torch.float32, device=device)
global_fit_param_stdd_t = torch.tensor(global_fit_param_stdd, dtype=torch.float32, device=device)

assert global_fit_params_t.numel() == Nmodes
assert global_fit_param_stdd_t.numel() == Nmodes

# Precompute normalized global spectrum for shape-only prior
g = global_fit_params_t.clamp_min(eps)
g_hat = g / (g.sum() + eps)
std = global_fit_param_stdd_t.clamp_min(eps)

# ------------------------
# Loss: Gaussian NLL for correlation-shape + shape-only prior
# ------------------------
def gaussian_loss_corrshape(
    y_field: torch.Tensor,         # [N,N], float32, normalized to unit variance
    log_params: torch.Tensor,       # [Nmodes], float32 (unconstrained)
) -> torch.Tensor:
    """
    y_field is the (demeaned, variance-normalized) residual image.
    log_params parameterize a positive spectrum s = exp(log_params).

    We build a *correlation* precision matrix using your basis form, and use:
       y^T C^{-1} y + logdet(C) + kappa * prior(shape)

    Prior is shape-only:
       s_hat = s / sum(s)
       penalize (s_hat - g_hat)^2 / std^2
    """
    # positive real spectrum params
    s = torch.exp(log_params).clamp_min(eps)    # [Nmodes]

    # shape-only prior
    s_hat = s / (s.sum() + eps)
    prior = ((s_hat - g_hat)**2 / (std**2 + eps)).sum()

    # convert s to complex for typing in diag
    s_c = torch.complex(s, torch.zeros_like(s))

    # precision matrix via basis (same structure you used)
    # C^{-1} = B diag(1/s) B*
    inv_diag = torch.complex(1.0 / s, torch.zeros_like(s))
    cov_inv = B @ torch.diag(inv_diag) @ B_conjT   # [Npix, Npix], complex

    # y_diff = 0 - y = -y
    y = y_field.reshape(-1)                         # [Npix]
    y_c = torch.complex(y, torch.zeros_like(y))     # complex

    # quadratic term: y^T C^{-1} y
    # (y^H cov_inv y) since y real; take real part
    quad = torch.real(torch.conj(y_c) @ (cov_inv @ y_c))

    # logdet term (approximate in your basis parameterization)
    # For a true diagonalization, logdet(C) = sum(log s) + const.
    # Keep your original form but now it's acting on correlation scale (variance already fixed by normalization).
    logdet = torch.sum(torch.log(s))

    return quad + logdet + kappa * prior

# ------------------------
# Simple minimizer (per image)
# ------------------------
def minimize_for_image(y_field: torch.Tensor, epochs: int, lr: float) -> torch.Tensor:
    # init at zeros => s = exp(0)=1
    params = torch.zeros((Nmodes,), dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([params], lr=lr)

    for ep in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = gaussian_loss_corrshape(y_field=y_field, log_params=params)
        loss.backward()
        opt.step()
        if ep % 25 == 0 or ep == epochs - 1:
            print(f"  ep={ep:4d}  loss={loss.item():.6f}")
    return params.detach()

# ------------------------
# Fit all images
# ------------------------
fit_params_shape = np.zeros((n_img, Nmodes), dtype=np.float32)

for img_idx in range(n_img):
    print(f"\nFitting image {img_idx+1}/{n_img}")
    y_field = err_fields_tensor[img_idx]  # [N,N], already demeaned + unit-variance
    log_params_hat = minimize_for_image(y_field, epochs=epochs, lr=lr)
    fit_params_shape[img_idx] = torch.exp(log_params_hat).cpu().numpy()

# Save:
# - correlation-shape spectra (unit-variance fit)
# - per-image variance (for re-injection at reconstruction time)
np.save(os.path.join(out_dir, "stage2_corrshape_spectra.npy"), fit_params_shape)
np.save(os.path.join(out_dir, "stage2_img_var.npy"), img_var.astype(np.float32))
np.save(os.path.join(out_dir, "stage2_img_mean.npy"), img_mean.squeeze().astype(np.float32))

print("\nSaved:")
print("  stage2_corrshape_spectra.npy  (spectrum for correlation shape)")
print("  stage2_img_var.npy            (per-image variance to scale covariance)")
print("  stage2_img_mean.npy           (per-image mean removed before fitting)")
