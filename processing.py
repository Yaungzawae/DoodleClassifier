import numpy as np
from scipy.ndimage import affine_transform, zoom

def random_augment_image(imgs_flat,
                         scale_range=(0.9, 1.1),   # Scale range for zooming
                         rotation_range=(-15, 15), # Rotation range in degrees
                         offset_range=(-3, 3),     # Offset range for translation
                         noise_std_range=(0, 2.5)):# Standard deviation range for Gaussian noise
    """
    Applies random affine transform + noise + 2px padding (result stays 28x28).
    imgs_flat: (N, 784)
    returns: (N, 784)
    """
    N = imgs_flat.shape[0]
    imgs = imgs_flat.reshape(N, 28, 28).astype(np.float32)

    # Generate random parameters
    scales = np.random.uniform(*scale_range, size=N)
    angles = np.deg2rad(np.random.uniform(*rotation_range, size=N))
    shifts_y = np.random.uniform(*offset_range, size=N)
    shifts_x = np.random.uniform(*offset_range, size=N)
    noise_stds = np.random.uniform(*noise_std_range, size=N)

    center = 13.5  

    augmented = np.empty((N, 28, 28), dtype=np.float32)

    for i in range(N):
        s = scales[i]
        c, si = np.cos(angles[i]), np.sin(angles[i])
        matrix = np.array([[c, -si], [si, c]]) / s

        offset = np.array([center, center]) - matrix @ np.array([center - shifts_x[i], center - shifts_y[i]])
        transformed = affine_transform(imgs[i], matrix, offset=offset, order=1, mode='constant', cval=0)

        # Crop center 24x24
        cropped = transformed[2:26, 2:26]

        # Add noise if needed
        if noise_stds[i] > 0:
            noise = np.random.normal(0, noise_stds[i], cropped.shape)
            cropped += noise

        # Pad back to 28x28 with 2px margin
        padded = np.pad(np.clip(cropped, 0, 255), pad_width=2, mode='constant', constant_values=0)
        augmented[i] = padded

    return augmented.reshape(N, 784)
