import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.utils import save_image
from datasets import ChestXrayPairDataset
from tqdm import tqdm

# ─── Generator: ConvVAE ─────────────────────────────────────────
class ConvVAE(nn.Module):
    def __init__(self, z_dim=256, in_channels=1, base_channels=32, img_size=1024):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, base_channels, 4, 2, 1), nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.Conv2d(base_channels, base_channels*2, 4, 2, 1), nn.ReLU(True))
        self.enc3 = nn.Sequential(nn.Conv2d(base_channels*2, base_channels*4, 4, 2, 1), nn.ReLU(True))
        self.enc4 = nn.Sequential(nn.Conv2d(base_channels*4, base_channels*8, 4, 2, 1), nn.ReLU(True))
        self.enc5 = nn.Sequential(nn.Conv2d(base_channels*8, base_channels*8, 4, 2, 1), nn.ReLU(True))
        h = img_size // (2**5)
        C = base_channels*8
        feat_dim = C * h * h
        self.fc_mu     = nn.Linear(feat_dim, z_dim)
        self.fc_logvar = nn.Linear(feat_dim, z_dim)
        self.fc_dec    = nn.Linear(z_dim, feat_dim)
        self.dec5 = nn.Sequential(nn.ConvTranspose2d(C, C, 4, 2, 1), nn.ReLU(True))
        self.dec4 = nn.Sequential(nn.ConvTranspose2d(C + base_channels*8, base_channels*4, 4, 2, 1), nn.ReLU(True))
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(base_channels*4*2, base_channels*2, 4, 2, 1), nn.ReLU(True))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(base_channels*2*2, base_channels, 4, 2, 1), nn.ReLU(True))
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(base_channels*2, in_channels, 4, 2, 1), nn.Sigmoid())

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        flat = e5.view(e5.size(0), -1)
        mu    = self.fc_mu(flat)
        logvar= self.fc_logvar(flat)
        return mu, logvar, (e1, e2, e3, e4)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, skips):
        e1, e2, e3, e4 = skips
        batch = z.size(0)
        feat  = self.fc_dec(z)
        C = self.dec5[0].in_channels
        h = int((feat.size(1) / C) ** 0.5)
        d5 = feat.view(batch, C, h, h)
        d5 = self.dec5(d5)
        d4 = torch.cat([d5, e4], 1); d4 = self.dec4(d4)
        d3 = torch.cat([d4, e3], 1); d3 = self.dec3(d3)
        d2 = torch.cat([d3, e2], 1); d2 = self.dec2(d2)
        d1 = torch.cat([d2, e1], 1); out = self.dec1(d1)
        return out

    def forward(self, x):
        mu, logvar, skips = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, skips)
        return x_recon, mu, logvar

# ─── Discriminator ─────────────────────────────────────────────
class Discriminator(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        layers = []
        c = in_channels
        for mult in [1, 2, 4, 8]:
            layers += [nn.Conv2d(c, base_channels * mult, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
            c = base_channels * mult
        layers += [nn.Conv2d(c, 1, 4, 1, 0)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).view(x.size(0), -1)

# ─── PerceptualLoss (handles 1-channel input) ──────────────────
class PerceptualLoss(nn.Module):
    def __init__(self, layer_ids=[3, 8, 15]):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.eval()
        for p in vgg.parameters(): p.requires_grad = False
        self.slices = []
        prev = 0
        for idx in layer_ids:
            self.slices.append(nn.Sequential(*list(vgg.children())[prev:idx]))
            prev = idx
        self.slices = nn.ModuleList(self.slices)
        self.criterion = nn.L1Loss()

    def forward(self, recon, target):
        if recon.size(1) == 1:
            recon  = recon.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        loss = 0
        x, re_y = recon, target
        for slice in self.slices:
            x    = slice(x)
            re_y = slice(re_y)
            loss += self.criterion(x, re_y)
        return loss

# ─── Training Loop ────────────────────────────────────────────
def train_vae_gan():
    img_size, batch_size = 1024, 4
    lr, z_dim = 1e-4, 256
    num_epochs = 50

    device = torch.device(
        'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    ds = ChestXrayPairDataset(
        "./BS_dataset_split/train",
        img_size,
        transform
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    G = ConvVAE(z_dim, 1, 32, img_size).to(device)
    D = Discriminator(1, 32).to(device)
    P = PerceptualLoss().to(device)

    opt_G = optim.Adam(G.parameters(), lr=lr)
    opt_D = optim.Adam(D.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    os.makedirs('samples', exist_ok=True)

    best_g_loss = float('inf')
    for ep in range(num_epochs):
        epoch_desc = f"Epoch {ep+1}/{num_epochs}"
        for x, y, _ in tqdm(loader, desc=epoch_desc, leave=False):
            x, y = x.to(device), y.to(device)

            recon, mu, lv = G(x)
            real_logits = D(y)
            fake_logits = D(recon.detach())
            d_loss = (
                bce(real_logits, torch.ones_like(real_logits)) +
                bce(fake_logits, torch.zeros_like(fake_logits))
            )
            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            # Generator update
            recon, mu, lv = G(x)
            fake_logits = D(recon)
            recon_l = nn.functional.l1_loss(recon, y)
            kld     = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
            gan_l   = bce(fake_logits, torch.ones_like(fake_logits))
            perc_l  = P(recon, y)
            g_loss  = recon_l + 0.01 * kld + 0.5 * gan_l + 0.1 * perc_l
            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

        # save samples at epoch end
        save_image(recon[:4], f'samples/epoch{ep+1}.png', nrow=2)
        save_image(y[:4],  f'samples/gt{ep+1}.png',    nrow=2)
        print(f"Epoch {ep+1}/{num_epochs} | D_loss {d_loss.item():.4f} G_loss {g_loss.item():.4f}")

        # checkpoint best models
        if g_loss.item() < best_g_loss:
            best_g_loss = g_loss.item()
            os.makedirs('models', exist_ok=True)
            torch.save(G.state_dict(), 'models/best_G.pth')
            torch.save(D.state_dict(), 'models/best_D.pth')
            print(f"[Checkpoint] Saved best models at epoch {ep+1}, G_loss {best_g_loss:.4f}")

if __name__ == '__main__':
    train_vae_gan()