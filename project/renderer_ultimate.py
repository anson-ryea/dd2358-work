import h5py as h5
import numpy as np
import torch
import torch.nn.functional as fun
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

def transfer_function_ultimate(x):
    """Vectorized transfer function running natively on the Metal GPU."""
    # Pre-calculating the squared differences speeds up the element-wise operations
    x_minus_9_sq = (x - 9.0) ** 2
    x_minus_3_sq = (x - 3.0) ** 2
    x_plus_3_sq  = (x + 3.0) ** 2

    r = 1.0 * torch.exp(-x_minus_9_sq / 1.0) + 0.1 * torch.exp(-x_minus_3_sq / 0.1) + 0.1 * torch.exp(-x_plus_3_sq / 0.5)
    g = 1.0 * torch.exp(-x_minus_9_sq / 1.0) + 1.0 * torch.exp(-x_minus_3_sq / 0.1) + 0.1 * torch.exp(-x_plus_3_sq / 0.5)
    b = 0.1 * torch.exp(-x_minus_9_sq / 1.0) + 0.1 * torch.exp(-x_minus_3_sq / 0.1) + 1.0 * torch.exp(-x_plus_3_sq / 0.5)
    a = 0.6 * torch.exp(-x_minus_9_sq / 1.0) + 0.1 * torch.exp(-x_minus_3_sq / 0.1) + 0.01 * torch.exp(-x_plus_3_sq / 0.5)

    return r, g, b, a

def save_image_worker_ultimate(args):
    """Background thread worker to keep Matplotlib off the main CPU thread."""
    img_array, index = args
    mpimg.imsave(f"img_out/volumerender{index}.png", img_array)

def render_ultimate(path, N, Nangles):
    """Ultimate Batch-Optimized Renderer for Apple Silicon (MPS)."""
    os.makedirs("img_out", exist_ok=True)

    # 1. Enforce MPS Device
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available! Ensure you are running on an Apple Silicon Mac with PyTorch natively installed.")
    device = torch.device("mps")

    # 2. Load Datacube directly to MPS
    with h5.File(path, "r") as f:
        datacube = np.array(f["density"])

    volume = torch.tensor(datacube, dtype=torch.float32, device=device)
    volume = volume.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, Nx, Ny, Nz)

    # 3. Dynamic Memory Batching (Crucial for Mac Unified Memory)
    # A 5D grid_sample request can easily consume 20GB+ of RAM if N is large.
    # We estimate the memory footprint and chunk the Nangles to keep it under ~2GB per pass.
    bytes_per_voxel = 4 # float32
    memory_per_angle = (N ** 3) * bytes_per_voxel * 3 # 3 channels roughly
    safe_ram_limit = 2 * (1024 ** 3) # 2 GB limit per batch

    batch_size = max(1, safe_ram_limit // memory_per_angle)
    # Make sure we don't have a batch size larger than our total angles
    batch_size = min(batch_size, Nangles)

    # Base Camera Grid
    c = torch.linspace(-1, 1, N, device=device)
    qy, qx, qz = torch.meshgrid(c, c, c, indexing='ij')

    # 4. Process in safe batches
    rendered_images_cpu = []

    for start_idx in range(0, Nangles, batch_size):
        end_idx = min(start_idx + batch_size, Nangles)
        current_batch_size = end_idx - start_idx

        # Generate exactly the angles needed for this batch
        indices = torch.arange(start_idx, end_idx, device=device)
        angles = (np.pi / 2 * indices / Nangles).view(-1, 1, 1, 1)

        # Broadcast rotations across the batch
        qyR = qy.unsqueeze(0) * torch.cos(angles) - qz.unsqueeze(0) * torch.sin(angles)
        qzR = qy.unsqueeze(0) * torch.sin(angles) + qz.unsqueeze(0) * torch.cos(angles)
        qxR = qx.unsqueeze(0).expand(current_batch_size, -1, -1, -1)

        # Stack into grid shape: (Batch, N, N, N, 3)
        grid = torch.stack((qzR, qyR, qxR), dim=-1)

        # Expand the volume to match the batch size
        volume_batched = volume.expand(current_batch_size, -1, -1, -1, -1)

        #
        # Massive parallel interpolation on the Metal GPU
        camera_grid = fun.grid_sample(
            volume_batched,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        ).squeeze(1) # Remove channel dim -> (Batch, N, N, N)

        # Calculate logs for the whole batch
        log_grid = torch.log(camera_grid.clamp(min=1e-10))

        # Alpha Compositing Loop (Running vectorized on MPS)
        image = torch.zeros((current_batch_size, N, N, 3), device=device)

        for j in range(N):
            dataslice = log_grid[:, j, :, :] # Sliced across depth for all angles
            r, g, b, a = transfer_function_ultimate(dataslice)

            image[..., 0] = a * r + (1 - a) * image[..., 0]
            image[..., 1] = a * g + (1 - a) * image[..., 1]
            image[..., 2] = a * b + (1 - a) * image[..., 2]

        image = torch.clamp(image, 0.0, 1.0)

        # Move this finalized batch back to CPU RAM immediately to free up GPU memory
        rendered_images_cpu.append(image.cpu().numpy())

    # Flatten the list of batches into a single array
    final_images = np.concatenate(rendered_images_cpu, axis=0)

    # 5. Asynchronous Multi-threaded I/O
    print("GPU math complete. Dispatching save tasks to CPU threads...")
    save_tasks = [(final_images[i], i) for i in range(Nangles)]

    with ThreadPoolExecutor() as executor:
        executor.map(save_image_worker_ultimate, save_tasks)

    # Projection Fallback
    plt.imsave("img_out/projection.png", np.log(np.mean(datacube, 0)), cmap="viridis", vmin=-5, vmax=5)
