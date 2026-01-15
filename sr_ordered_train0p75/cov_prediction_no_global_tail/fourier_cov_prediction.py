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
# Model: Predict only low-k Fourier modes
# ============================================================
class FullSpectrumNet(nn.Module):
    def __init__(self, nk=33*33):
        super().__init__()
        self.nk = nk

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(16, 16, 5, padding=2),
            nn.ReLU(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.nk, 512),
            nn.ReLU(),
            nn.Linear(512, self.nk),
        )

    def forward(self, x):
        B = x.shape[0]

        feat = self.encoder(x)                    # (B,16,64,64)
        F = torch.fft.rfft2(feat, norm="ortho")   # (B,16,64,33)

        mag = torch.sqrt(F.real**2 + F.imag**2 + 1e-12)
        mag = mag.mean(dim=1)                     # (B,64,33)
        mag = mag[:, :33, :33]                    # (B,33,33)
        mag_flat = mag.reshape(B, -1)             # (B,1089)

        log_s_hat = self.mlp(mag_flat)
        log_s_hat = torch.clamp(log_s_hat, -20.0, 20.0)

        return log_s_hat



# ============================================================
# Dataset (low-k targets only)
# ============================================================
class SpectrumDataset(Dataset):
    def __init__(self, mean_fields, log_s):
        self.x = torch.tensor(mean_fields, dtype=torch.float32).unsqueeze(1)
        self.t = torch.tensor(log_s, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.t[idx]


# ============================================================
# Training
# ============================================================
def train_spectrum_predictor(
    mean_fields,
    log_s_train,
    n_epochs=40,
    batch_size=32,
    lr=5e-4,
):
    model = FullSpectrumNet(nk=log_s_train.shape[1]).to(device)

    ds = SpectrumDataset(mean_fields, log_s_train)
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

            log_s_hat = model(x_batch)
            loss = loss_fn(log_s_hat, t_batch)

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
    outfile_prefix="train",
):
    model.eval()
    ds = torch.tensor(mean_fields, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    preds = []

    with torch.no_grad():
        for x_batch in loader:
            x_batch = x_batch.to(device)
            log_s = model(x_batch).cpu().numpy()
            preds.append(log_s)

    log_s_all = np.concatenate(preds, axis=0)
    s_all = np.exp(log_s_all)

    np.save(f"{outfile_prefix}_predicted_log_s.npy", log_s_all)
    np.save(f"{outfile_prefix}_predicted_s.npy", s_all)

    print(f"Saved predictions for {outfile_prefix}:")
    print("  log-spectrum:", log_s_all.shape)
    print("  spectrum:", s_all.shape)



# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    test_mean_fields = np.load(
        "../stage_1/output_data/mse_5l_i123_c32s_padR_schLrG0p95_reg0_testPredFields.npy"
    )
    train_mean_fields = np.load(
        "../stage_1/output_data/mse_5l_i123_c32s_padR_schLrG0p95_reg0_trainPredFields.npy"
    )
    s_train = np.load(
        "../stage_2/output/parCov_fitting_fourier_allImages_mseStart_globalPrior_anal_empPriorStdOn0p5_param_fits.npy"
    )

    print("mean_fields shape:", train_mean_fields.shape)
    print("s_train shape:", s_train.shape)

    # log-spectrum targets
    s_train = np.maximum(s_train, 1e-12)
    log_s_train = np.log(s_train)

    print("log-spectrum stats:")
    print(" min:", log_s_train.min())
    print(" max:", log_s_train.max())

    model = train_spectrum_predictor(
        mean_fields=train_mean_fields,
        log_s_train=log_s_train,
        n_epochs=40,
        batch_size=32,
        lr=5e-4,
    )

    save_predictions(
        model=model,
        mean_fields=train_mean_fields,
        outfile_prefix="train",
    )

    save_predictions(
        model=model,
        mean_fields=test_mean_fields,
        outfile_prefix="test",
    )
