import sys
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Fix for importing from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vae_dataset import ShapeColorDataset
from vae_model import ConvVAE

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
BATCH_SIZE = 32
LATENT_DIM = 32
SAMPLE_EVERY = 10

# Output folders
os.makedirs("models", exist_ok=True)
os.makedirs("samples/joint", exist_ok=True)

# DataLoader
dataset = ShapeColorDataset("data/toy_dataset/train")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model + optimizer
model = ConvVAE(latent_dim=LATENT_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# VAE loss
def vae_loss(x_recon, x, mu, logvar):
    recon = F.mse_loss(x_recon, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x in dataloader:
        x = x.to(DEVICE)
        x_recon, mu, logvar = model(x)
        loss = vae_loss(x_recon, x, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {total_loss:.2f}")

    # Save visual outputs
    if (epoch + 1) % SAMPLE_EVERY == 0 or epoch == EPOCHS - 1:
        model.eval()
        with torch.no_grad():
            recon, _, _ = model(x[:8])
            save_image(x[:8], f"samples/joint/input_epoch{epoch+1}.png", nrow=4)
            save_image(recon, f"samples/joint/output_epoch{epoch+1}.png", nrow=4)
        model.train()

# Save model
torch.save(model.state_dict(), "models/joint_vae.pth")
print(" Joint VAE model saved.")
