import os
import torch
import torchvision.utils as vutils
from VAE_GAN import ConvVAE
from datasets import ChestXrayPairDataset
from torchvision import transforms
from torch.utils.data import DataLoader


def test_vae_gan(
    model_path: str,
    test_root: str,
    output_dir: str,
    img_size: int = 1024,
    batch_size: int = 4,
    z_dim: int = 256,
    base_ch: int = 32,
    num_workers: int = 4,
    max_batches: int = None
):
    """
    ConvVAE-GAN 테스트 함수: filename 반환하도록 변경
    """
    os.makedirs(output_dir, exist_ok=True)

    device = (
        torch.device('mps') if (torch.backends.mps.is_available() and torch.backends.mps.is_built())
        else torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu')
    )
    print(f"Using device: {device}")

    # Generator 모델 불러오기
    model = ConvVAE(z_dim=z_dim, in_channels=1, base_channels=base_ch, img_size=img_size).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Dataset & DataLoader
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    test_ds = ChestXrayPairDataset(root_dir=test_root, img_size=img_size, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"[DEBUG] test dataset size: {len(test_ds)}")
    print(f"[DEBUG] filenames: {test_ds.filenames}")

    # 테스트 및 이미지 저장
    with torch.no_grad():
        for batch_idx, (x, y, fnames) in enumerate(test_loader):
            x = x.to(device)
            x_recon, mu, logvar = model(x)

            for i in range(x.size(0)):
                # 확장자 제거
                name, _ = os.path.splitext(fnames[i])
                vutils.save_image(
                    x_recon[i], os.path.join(output_dir, f"recon_{name}.png")
                )
                vutils.save_image(
                    y[i].to(device), os.path.join(output_dir, f"gt_{name}.png")
                )
                vutils.save_image(
                    x[i], os.path.join(output_dir, f"orig_{name}.png")
                )

            if max_batches is not None and batch_idx + 1 >= max_batches:
                break

    print(f"Saved reconstruction samples to {output_dir}")


if __name__ == "__main__":
    model_path = "./models_GAN_1024_50epoch/best_G.pth"
    test_root  = "./BS_dataset_split/test"
    output_dir = "./outputs"

    test_vae_gan(
        model_path=model_path,
        test_root=test_root,
        output_dir=output_dir,
        img_size=1024,
        batch_size=4,
        z_dim=256,
        base_ch=32,
        num_workers=4,
        max_batches=None
    )