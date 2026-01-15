import numpy as np
import os

# ============================================================
# Configuration
# ============================================================
model_identifier_str = "0p5"   # e.g. "0p5"
dataset_label = 'train'

N = 64
hr_data_size = 64
n_samples = 100

outdir = "."
os.makedirs(outdir, exist_ok=True)

# ============================================================
# FFT-based noise generation
# ============================================================
def get_noise_from_cov_params_fft(cov_params, n_samples=10, N=64):
    """
    Generate spatial noise realizations from Fourier-diagonal covariance.

    Parameters
    ----------
    cov_params : array, shape (1089,) or (33,33)
        Fourier power spectrum (rfft2 layout).
    n_samples : int
        Number of noise realizations.
    N : int
        Grid size (NxN).

    Returns
    -------
    noise : ndarray, shape (n_samples, N*N)
        Spatial noise samples.
    Zk : ndarray, shape (n_samples, 33, 33)
        Standard normal Fourier coefficients (complex).
    """

    # reshape to rfft2 half-plane
    s_k = cov_params.reshape(33, 33)

    # draw complex Gaussian Fourier coefficients
    Zr = np.random.randn(n_samples, 33, 33)
    Zi = np.random.randn(n_samples, 33, 33)
    Zk = (Zr + 1j * Zi) / np.sqrt(2.0)

    # scale by sqrt spectrum
    Fk = Zk * np.sqrt(s_k)[None, :, :]

    # inverse FFT to real space
    noise = np.fft.irfft2(Fk, s=(N, N), norm="ortho")

    return noise.reshape(n_samples, N * N), Zk


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    # --------------------------------------------------------
    # Load image-specific Fourier covariance parameters
    # --------------------------------------------------------
    cov_param_file = (
        f"../stage_2/output/parCov_fitting_fourier_allImages_"
        f"mseStart_globalPrior_anal_empPriorStdOn{model_identifier_str}_"
        f"param_fits.npy"
    )

    cov_params_all = np.load(cov_param_file)

    print("Loaded covariance params:", cov_params_all.shape)
    print("Generating noise samples per image:", n_samples)

    # safety: avoid sqrt of zero or negative
    cov_params_all = np.maximum(cov_params_all, 0.0)

    # --------------------------------------------------------
    # Generate noise per image
    # --------------------------------------------------------
    for i in range(len(cov_params_all)):

        noise, Zk = get_noise_from_cov_params_fft(
            cov_params_all[i],
            n_samples=n_samples,
            N=N
        )

        np.save(
            os.path.join(outdir, f"p{model_identifier_str}_{dataset_label}_img_{i}_noise_samples.npy"),
            noise
        )
        np.save(
            os.path.join(outdir, f"p{model_identifier_str}_{dataset_label}_img_{i}_noise_Zk.npy"),
            Zk
        )

        if i % 25 == 0:
            print(f"Generated noise for image {i}")

    print("Done.")
