
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as fun
import os

def transfer_torch(x):
    # Converted to torch functions to keep data on the GPU
    r = (
        1.0 * torch.exp(-((x - 9.0) ** 2) / 1.0)
        + 0.1 * torch.exp(-((x - 3.0) ** 2) / 0.1)
        + 0.1 * torch.exp(-((x - -3.0) ** 2) / 0.5)
    )
    g = (
        1.0 * torch.exp(-((x - 9.0) ** 2) / 1.0)
        + 1.0 * torch.exp(-((x - 3.0) ** 2) / 0.1)
        + 0.1 * torch.exp(-((x - -3.0) ** 2) / 0.5)
    )
    b = (
        0.1 * torch.exp(-((x - 9.0) ** 2) / 1.0)
        + 0.1 * torch.exp(-((x - 3.0) ** 2) / 0.1)
        + 1.0 * torch.exp(-((x - -3.0) ** 2) / 0.5)
    )
    a = (
        0.6 * torch.exp(-((x - 9.0) ** 2) / 1.0)
        + 0.1 * torch.exp(-((x - 3.0) ** 2) / 0.1)
        + 0.01 * torch.exp(-((x - -3.0) ** 2) / 0.5)
    )

    return r, g, b, a

def render_torch(path, N, Nangles):
    """Volume Rendering Accelerated via PyTorch"""

    # Automatically fallback to CPU if no GPU is available, but GPU is highly recommended
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs("img_out", exist_ok=True)

    # 1. Load Datacube
    with h5.File(path, "r") as f:
        datacube = np.array(f["density"])

    # 2. Prepare tensor for grid_sample: shape (1, 1, Nx, Ny, Nz)
    # grid_sample expects a 5D tensor for 3D volumes (Batch, Channels, Depth, Height, Width)
    volume = torch.tensor(datacube, dtype=torch.float32, device=device)
    volume = volume.unsqueeze(0).unsqueeze(0)

    # 3. Base Camera Grid (normalized [-1, 1] for grid_sample)
    # Using 'ij' indexing with swapped outputs to accurately mimic numpy's default 'xy' meshgrid behaviour
    c = torch.linspace(-1, 1, N, device=device)
    qy, qx, qz = torch.meshgrid(c, c, c, indexing='ij')

    # Do Volume Rendering at Different Viewing Angles
    for i in range(Nangles):
        angle = np.pi / 2 * i / Nangles

        # Rotate camera view (pure tensor operations)
        qxR = qx
        qyR = qy * np.cos(angle) - qz * np.sin(angle)
        qzR = qy * np.sin(angle) + qz * np.cos(angle)

        # 4. Interpolate onto Camera Grid
        # grid_sample expects coordinates in (w, h, d) order corresponding to (z, y, x) spatial dimensions
        grid = torch.stack((qzR, qyR, qxR), dim=-1).unsqueeze(0)

        camera_grid = fun.grid_sample(
            volume,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        )

        # Squeeze batch and channel dims -> shape (N, N, N)
        camera_grid = camera_grid.squeeze(0).squeeze(0)

        # Pre-calculate logs for the entire grid to avoid recalculating in the loop
        log_grid = torch.log(camera_grid.clamp(min=1e-10))

        # 5. Do Volume Rendering (Alpha Compositing)
        image = torch.zeros((N, N, 3), device=device)

        # We iterate over the first dimension (depth slices)
        for j in range(N):
            dataslice = log_grid[j]
            r, g, b, a = transfer_torch(dataslice)

            image[:, :, 0] = a * r + (1 - a) * image[:, :, 0]
            image[:, :, 1] = a * g + (1 - a) * image[:, :, 1]
            image[:, :, 2] = a * b + (1 - a) * image[:, :, 2]

        image = torch.clamp(image, 0.0, 1.0).cpu().numpy()

        # 6. Plot & Save Volume Rendering
        plt.figure(figsize=(4, 4), dpi=80)
        plt.imshow(image)
        plt.axis("off")
        plt.savefig(
            f"img_out/volumerender{i}.png", dpi=240, bbox_inches="tight", pad_inches=0
        )
        plt.close('all')

    # Plot Simple Projection -- for Comparison
    plt.figure(figsize=(4, 4), dpi=80)
    plt.imshow(np.log(np.mean(datacube, 0)), cmap="viridis")
    plt.clim(-5, 5)
    plt.axis("off")
    plt.savefig("img_out/projection.png", dpi=240, bbox_inches="tight", pad_inches=0)
    plt.close('all')
