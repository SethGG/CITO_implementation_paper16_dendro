"""
CITO 24-25
Daniël Zee (s2063131) and Martijn Combé (s2599406)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from skimage import transform, util
from scipy.ndimage import gaussian_filter


def generate_wood_block(seed, resolution=(800, 600),
                        depth=2000,
                        early_wood_width_range=(6, 12),
                        early_wood_gray_range=(175, 200),
                        late_wood_width_range=(6, 12),
                        late_wood_gray_range=(50, 75),
                        rot_deg_height=0,
                        rot_deg_width=0,
                        gaussian_blur=2,
                        center_pith=False):
    """
    Generates a synthetic 3D tree-ring phantom of a wood block.

    Parameters:
        seed: random seed for reproducibility
        resolution: (width, height) of each 2D slice
        depth: number of slices (z-dimension)
        early_wood_width_range / gray_range: visual properties of earlywood rings
        late_wood_width_range / gray_range: visual properties of latewood rings
        rot_deg_height: rotation around the width axis (tilts ring height)
        rot_deg_width: rotation around the height axis (tilts ring width)
        gaussian_blur: sigma value for optional smoothing
        center_pith: if True, places the tree center in the image center

    Returns:
        image3d: 3D NumPy array (depth, height, width)
        slices_c: list of center coordinates (x, y) per slice
    """
    np.random.seed(seed)

    # Unpack dimensions
    width, height = resolution

    # Compute size corrections for height rotation (tilt in Y-Z)
    rot_rad_height = np.deg2rad(rot_deg_height)
    b1_height = np.tan(rot_rad_height) * depth
    b_height = height + b1_height
    d_height = np.cos(rot_rad_height) * b_height

    resize_height_for_height = int(d_height)
    resize_width_for_height = int(d_height / height * width)
    stretch_height = int(b_height)

    # Compute size corrections for width rotation (tilt in X-Z)
    rot_rad_width = np.deg2rad(rot_deg_width)
    b1_width = np.tan(rot_rad_width) * depth
    b_width = width + b1_width
    d_width = np.cos(rot_rad_width) * b_width

    resize_height_for_width = int(d_width / width * height)
    resize_width_for_width = int(d_width)
    stretch_width = int(b_width)

    # Determine final image size
    final_resize_height = max(resize_height_for_height, resize_height_for_width)
    final_resize_width = max(resize_width_for_height, resize_width_for_width)
    final_stretch_height = final_resize_height / np.cos(rot_rad_height)
    final_stretch_width = final_resize_width / np.cos(rot_rad_width)

    # Initialize grayscale image
    image = np.zeros((final_resize_height, final_resize_width), dtype=np.uint8)

    # Determine ring center
    if center_pith:
        cx = final_resize_width // 2
        cy = final_resize_height // 2
    else:
        cx = np.random.randint(final_resize_width)
        cy = np.random.randint(final_resize_height)

    # Initial ring size (radius)
    ring_size = int(np.sqrt((final_resize_width + final_resize_height) ** 2))

    # Start with either earlywood or latewood
    use_early_wood = np.random.choice([True, False])

    # Draw concentric rings
    while ring_size > 0:
        # Choose ring properties
        if use_early_wood:
            min_w, max_w = early_wood_width_range
            min_g, max_g = early_wood_gray_range
        else:
            min_w, max_w = late_wood_width_range
            min_g, max_g = late_wood_gray_range

        ring_width = np.random.randint(min_w, max_w + 1)
        ring_gray = np.random.randint(min_g, max_g + 1)

        # Define square region around current ring
        x_min = max(0, cx - ring_size)
        x_max = min(final_resize_width, cx + ring_size)
        y_min = max(0, cy - ring_size)
        y_max = min(final_resize_height, cy + ring_size)

        y_idx, x_idx = np.ogrid[y_min:y_max, x_min:x_max]
        distance_sq = (x_idx - cx) ** 2 + (y_idx - cy) ** 2
        mask = distance_sq <= ring_size ** 2

        # Apply ring color inside mask
        image[y_min:y_max, x_min:x_max][mask] = ring_gray

        ring_size -= ring_width
        use_early_wood = not use_early_wood

    # Apply affine stretching due to tilt
    if final_resize_height < final_stretch_height or final_resize_width < final_stretch_width:
        image = util.img_as_ubyte(transform.resize(image, (final_stretch_height, final_stretch_width)))

    # Apply blur to smooth transitions
    if gaussian_blur > 0:
        image = util.img_as_ubyte(gaussian_filter(image, sigma=gaussian_blur))

    # Extract 2D slices to form 3D volume
    image3d = np.array([
        image[
            int((final_stretch_height - stretch_height) / 2) + int(i * (stretch_height - height) / depth):
            int((final_stretch_height - stretch_height) / 2) + int(height + i * (stretch_height - height) / depth),
            int((final_stretch_width - stretch_width) / 2) + int(i * (stretch_width - width) / depth):
            int((final_stretch_width - stretch_width) / 2) + int(width + i * (stretch_width - width) / depth)
        ]
        for i in range(depth)
    ])

    # Compute center coordinates for each slice
    cx_stretched = int(cx * final_stretch_width / final_resize_width)
    cy_stretched = int(cy * final_stretch_height / final_resize_height)
    offset_x = int((final_stretch_width - stretch_width) / 2)
    offset_y = int((final_stretch_height - stretch_height) / 2)
    slices_cx = np.array([cx_stretched - int(i * (stretch_width - width) / depth) - offset_x for i in range(depth)])
    slices_cy = np.array([cy_stretched - int(i * (stretch_height - height) / depth) - offset_y for i in range(depth)])

    slices_c = list(zip(slices_cx, slices_cy))

    return image3d, slices_c


def interactive_slice_viewer(image3d):
    """
    Displays a GUI to scroll through 3D slices interactively.

    Axes:
        - Front-to-Back (axial): z-axis
        - Top-to-Bottom (coronal): y-axis
        - Left-to-Right (sagittal): x-axis
    """
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.3)

    current_axis = 0  # 0 = axial, 1 = coronal, 2 = sagittal
    slice_index = 0
    max_slices = [image3d.shape[0] - 1, image3d.shape[1] - 1, image3d.shape[2] - 1]
    dimensions = [(image3d.shape[1], image3d.shape[2]),
                  (image3d.shape[2], image3d.shape[0]),
                  (image3d.shape[1], image3d.shape[0])]

    slider_ax = plt.axes([0.42, 0.1, 0.5, 0.05])
    radio_ax = plt.axes([0.01, 0.01, 0.25, 0.2])
    radio_buttons = RadioButtons(radio_ax, ('Front-to-Back', 'Top-to-Bottom', 'Left-to-Right'))

    slice_slider, img_display = None, None

    def update_slice(val):
        """Update image when slider is moved."""
        slice_idx = int(slice_slider.val)
        if current_axis == 0:
            img_display.set_data(image3d[slice_idx, :, :])
        elif current_axis == 1:
            img_display.set_data(image3d[:, slice_idx, :].T)
        elif current_axis == 2:
            img_display.set_data(image3d[:, :, slice_idx].T)

    def update_axis(label):
        """Switch axis view and reset slider."""
        nonlocal current_axis, img_display, slice_slider

        current_axis = {'Front-to-Back': 0, 'Top-to-Bottom': 1, 'Left-to-Right': 2}[label]

        slider_ax.clear()
        slice_slider = Slider(
            ax=slider_ax,
            label='Slice Index',
            valmin=0,
            valmax=max_slices[current_axis],
            valinit=slice_index,
            valstep=int(max_slices[current_axis]/20)+1
        )
        slice_slider.on_changed(update_slice)

        ax.clear()
        view = (image3d[0, :, :] if current_axis == 0
                else image3d[:, 0, :].T if current_axis == 1
                else image3d[:, :, 0].T)
        img_display = ax.imshow(view, cmap='gray')
        ax.set_xlim([0, dimensions[current_axis][1]])
        ax.set_ylim([dimensions[current_axis][0], 0])

    radio_buttons.on_clicked(update_axis)
    update_axis('Front-to-Back')
    plt.show()


# Example usage for testing the phantom and GUI
if __name__ == "__main__":
    image3d, _ = generate_wood_block(seed=2, rot_deg_height=7, rot_deg_width=7)
    interactive_slice_viewer(image3d)
