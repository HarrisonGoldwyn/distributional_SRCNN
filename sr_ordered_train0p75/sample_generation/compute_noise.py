import numpy as np

hr_data_size = 64
N = 64

def compl_dft_basis(x, y, k_x, k_y):
    return np.exp(1j * 2*np.pi * (k_x*x + k_y*y)/N)


if __name__ == "__main__":

    x = np.arange(0, N)
    xg = np.tile(x, N)            # generates x-coordinates for the entire grid N*K 
    yg = np.repeat(x, N, axis=0)
    mat_xg = xg.reshape((64, 64))
    mat_yg = yg.reshape((64, 64))


    ## Create basis functions to max period
    # max_T = 20
    # min_k = int(N/max_T)
    min_k = 0
    # max_k = (N)//2
    max_k = (N)//2 + 1

    basis_function_k_idx = []
    basis_functions = []
    for _kx in range(min_k, max_k):
        for _ky in range(min_k, max_k):
            basis_functions.append(
                compl_dft_basis(
                    mat_xg, 
                    mat_yg, 
                    _kx, 
                    _ky
                    )
                )
            basis_function_k_idx.append((_kx, _ky))
            
    basis_functions = np.asarray(basis_functions).reshape((-1, 64**2)).T / N
    basis_functions.shape

    def get_noise_from_cov_params_wSVD(cov_params, n_samples=10):

        Sigma = np.matmul(
            np.matmul(
                basis_functions, 
                np.diag(cov_params)
                ),
            np.conj(basis_functions.T)
            )
        
        u, s, vh = np.linalg.svd(np.real(Sigma))
        
        Z = np.random.randn(n_samples, hr_data_size**2)
        
        rescaled_noise = Z @ np.diag(s**0.5) @ u.T
        
        return rescaled_noise, Z

    ## analytic results with correct downscaling
    
    p5p5_cov_params = np.load(
        "/projects/ecrpstats/distributional_SRCNN/sr_ordered_train0p75/stage_2/" \
        "parCov_fitting_fourier_allImages_mseStart_globalPrior_anal_empPriorStdOn5p5_param_fits.npy")


    for i in range(len(p5p5_cov_params)):
        _noise, _Z = get_noise_from_cov_params_wSVD(p5p5_cov_params[i], n_samples=100)

        np.save(f'p2p5_img_{i}_noise_samples.npy', _noise)
        np.save(f'p2p5_img_{i}_noise_Z.npy', _Z)
