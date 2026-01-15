import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Device
# ============================================================
device = (
    "cuda"
    if torch.cuda.is_available()
    # else "mps"
    # if torch.backends.mps.is_available()
    else "cpu"
)
print("Using device:", device)


# ============================================================
# Build Low-k Mask
# ============================================================
def build_lowk_mask(k_max=10.0):
    """
    Construct a mask for |k| <= k_max over the 33x33 half-plane (kx,ky = 0..32).
    Returns:
        lowk_idx  : array of flattened indices into (33*33=1089)
        mask_flat : boolean mask (1089,)
    """
    N = 64
    kx = np.arange(0, N//2 + 1)  # 0..32
    ky = np.arange(0, N//2 + 1)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    R = np.sqrt(KX**2 + KY**2)

    mask = (R <= k_max)          # (33,33)
    mask_flat = mask.reshape(-1) # (1089,)
    lowk_idx = np.where(mask_flat)[0]

    print(f"Low-k modes: {len(lowk_idx)} of 1089")
    return lowk_idx, mask_flat


# ============================================================
# Model: Predict only low-k Fourier modes
# ============================================================
class LowKFullSpectrumNet(nn.Module):
    def __init__(self, lowk_idx):
        super().__init__()
        self.register_buffer("lowk_idx", torch.tensor(lowk_idx, dtype=torch.long))
        self.nk = len(lowk_idx)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(16, 16, 5, padding=2),
            nn.ReLU(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.nk, 256),
            nn.ReLU(),
            nn.Linear(256, self.nk)
        )

    def forward(self, x):
        B = x.shape[0]

        feat = self.encoder(x)                    # (B,16,64,64)
        F = torch.fft.rfft2(feat, norm="ortho")   # (B,16,64,33)
        mag = torch.sqrt(
            F.real**2 + F.imag**2                 # (B,16,64,33)
            + 1e-12                               # "standard stabilizer for FFT magnitude pipelines"
            )
        mag = mag.mean(dim=1)                     # (B,64,33)
        mag = mag[:, :33, :33]                    # (B,33,33)
        mag_flat = mag.reshape(B, -1)             # (B,1089)

        # select low-k subset
        mag_low = mag_flat[:, self.lowk_idx]      # (B, nk)

        log_s_hat_low = self.mlp(mag_low)
        log_s_hat_low = torch.clamp(log_s_hat_low, min=-20.0, max=20.0)

        return log_s_hat_low


# ============================================================
# Dataset (low-k targets only)
# ============================================================
class SpectrumDataset(Dataset):
    def __init__(self, mean_fields, t_low):
        self.x = torch.tensor(mean_fields, dtype=torch.float32).unsqueeze(1)
        self.t = torch.tensor(t_low, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.t[idx]


# ============================================================
# Training
# ============================================================
def train_spectrum_predictor(
    mean_fields,
    t_train_low,
    lowk_idx,
    n_epochs=40,
    batch_size=32,
    lr=5e-4
):
    model = LowKFullSpectrumNet(lowk_idx).to(device)

    ds = SpectrumDataset(mean_fields, t_train_low)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(n_epochs):
        model.train()
        running = 0.0
        steps = 0

        for x_batch, t_batch in loader:
            x_batch = x_batch.to(device)
            t_batch = t_batch.to(device)

            log_s_low = model(x_batch)
            loss = loss_fn(log_s_low, t_batch)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += loss.item()
            steps += 1

        print(f"Epoch {ep:03d} | Loss = {running/steps:.6f}")

    return model


# ============================================================
# Save predictions (reconstruct full 1089-mode spectrum)
# ============================================================
def save_predictions(
    model,
    mean_fields,
    lowk_idx,
    s_global,
    outfile_prefix="train"
):
    # log(global spectrum)
    log_s_global = np.log(np.maximum(s_global, 1e-12))  # (1089,)
    log_s_global_t = torch.tensor(log_s_global, dtype=torch.float32)

    model.eval()
    ds = torch.tensor(mean_fields, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    full_preds = []

    with torch.no_grad():
        for x_batch in loader:
            x_batch = x_batch.to(device)
            log_s_low = model(x_batch).cpu().numpy()  # (B, nk)

            B = log_s_low.shape[0]

            # start from global for all high-k modes
            log_s_full = np.tile(log_s_global, (B, 1))  # (B,1089)

            # overwrite low-k modes
            log_s_full[:, lowk_idx] = log_s_low

            full_preds.append(log_s_full)

    log_s_full_all = np.concatenate(full_preds, axis=0)
    s_full_all = np.exp(log_s_full_all)

    np.save(f"{outfile_prefix}_predicted_log_s.npy", log_s_full_all)
    np.save(f"{outfile_prefix}_predicted_s.npy", s_full_all)

    print(f"Saved predictions for {outfile_prefix}:")
    print("  log-spectrum:", log_s_full_all.shape)
    print("  spectrum:", s_full_all.shape)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":


    # ------------------------------
    # Load inputs
    # ------------------------------
    test_mean_fields = np.load("../stage_1/output_data/mse_5l_i123_c32s_padR_schLrG0p95_reg0_testPredFields.npy")
    train_mean_fields = np.load("../stage_1/output_data/mse_5l_i123_c32s_padR_schLrG0p95_reg0_trainPredFields.npy")
    s_train = np.load("../stage_2/output/parCov_fitting_fourier_allImages_mseStart_globalPrior_anal_empPriorStdOn0p7_param_fits.npy")

    print("mean_fields shape:", train_mean_fields.shape)
    print("s_train shape:", s_train.shape)

    # ------------------------------
    # Build low-k mask
    # ------------------------------
    lowk_idx, mask_flat = build_lowk_mask(k_max=10.0)  # adjust k_max as needed

    # ------------------------------
    # Prepare low-k targets
    # ------------------------------
    s_train = np.maximum(s_train, 1e-12)
    t_train = np.log(s_train)              # (642,1089)
    t_train_low = t_train[:, lowk_idx]     # (642, M)

    # Also prepare global spectrum for full reconstruction
    s_global = s_train.mean(axis=0)        # (1089,)

    # ------------------------------
    # Diagnosis
    # ------------------------------
    # s_train is (642, 1089)
    s_train = np.maximum(s_train, 1e-40)
    t_train = np.log(s_train)

    t_train_low = t_train[:, lowk_idx]  # shape (642, 90)

    print("low-k log-spectrum stats:")
    print(" min:", np.min(t_train_low))
    print(" max:", np.max(t_train_low))

    # count extreme values
    print("vals < -20:", np.sum(t_train_low < -20))
    print("vals < -30:", np.sum(t_train_low < -30))
    print("vals >  20:", np.sum(t_train_low > 20))
    print("vals >  30:", np.sum(t_train_low > 30))

    # count zeros (which would have been log(1e-40) artificially)
    print("num exactly 1e-40 before log:", np.sum(s_train[:,lowk_idx] == 1e-40))


    # ------------------------------
    # Train
    # ------------------------------
    model = train_spectrum_predictor(
        mean_fields=train_mean_fields,
        t_train_low=t_train_low,
        lowk_idx=lowk_idx,
        n_epochs=40,
        batch_size=32,
        lr=5e-4,
    )

    # ------------------------------
    # Save predictions on train data (GOF)
    # ------------------------------
    save_predictions(
        model=model,
        mean_fields=train_mean_fields,
        lowk_idx=lowk_idx,
        s_global=s_global,
        outfile_prefix="train",
    )

    # ------------------------------
    # Save predictions on test data
    # ------------------------------
    save_predictions(
        model=model,
        mean_fields=test_mean_fields,
        lowk_idx=lowk_idx,
        s_global=s_global,
        outfile_prefix="test",
    )