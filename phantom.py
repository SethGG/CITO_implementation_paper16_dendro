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
                        gaussian_blur=2):
    # fix seed
    np.random.seed(seed)

    # extract final width and height
    width, height = resolution

    # calculate required full image size for rotation
    rot_rad_height = np.deg2rad(rot_deg_height)
    b1_height = np.tan(rot_rad_height) * depth
    b_height = height + b1_height
    d_height = np.cos(rot_rad_height) * b_height

    resize_height_for_height = int(d_height)
    resize_width_for_height = int(d_height / height * width)
    stretch_height = int(b_height)

    rot_rad_width = np.deg2rad(rot_deg_width)
    b1_width = np.tan(rot_rad_width) * depth
    b_width = width + b1_width
    d_width = np.cos(rot_rad_width) * b_width

    resize_height_for_width = int(d_width / width * height)
    resize_width_for_width = int(d_width)
    stretch_width = int(b_width)

    final_resize_height = max(resize_height_for_height, resize_height_for_width)
    final_resize_width = max(resize_width_for_height, resize_width_for_width)
    final_stretch_height = final_resize_height / np.cos(rot_rad_height)
    final_stretch_width = final_resize_width / np.cos(rot_rad_width)

    # create image with resize dimensions
    image = np.zeros((final_resize_height, final_resize_width), dtype=np.uint8)

    # generate the center point of the tree rings
    cx = np.random.randint(final_resize_width)
    cy = np.random.randint(final_resize_height)

    # determine the initial maximum ring size
    ring_size = int(np.sqrt((final_resize_width + final_resize_height) ** 2))

    # randomly determine the wood type of the outermost ring
    use_early_wood = np.random.choice([True, False])

    while ring_size > 0:
        # set width and color bounds for wood types
        if use_early_wood:
            min_ring_width, max_ring_width = early_wood_width_range
            min_ring_gray, max_ring_grey = early_wood_gray_range
        else:
            min_ring_width, max_ring_width = late_wood_width_range
            min_ring_gray, max_ring_grey = late_wood_gray_range
        # randomly determine ring width and color
        ring_width = np.random.randint(min_ring_width, max_ring_width+1)
        ring_color = np.random.randint(min_ring_gray, max_ring_grey+1)

        # detect if the ring does outside of image bounds
        x_min = max(0, cx - ring_size)
        x_max = min(final_resize_width, cx + ring_size)
        y_min = max(0, cy - ring_size)
        y_max = min(final_resize_height, cy + ring_size)
        y_indices, x_indices = np.ogrid[y_min:y_max, x_min:x_max]
        # determine distance from each point to the circle center
        distances_sq = (x_indices - cx)**2 + (y_indices - cy)**2
        # create a mask for all point within the ring size
        mask = distances_sq <= ring_size**2
        # set all point within the ring size to the ring color
        image[y_min:y_max, x_min:x_max][mask] = ring_color

        # set the ringsize for the next ring to generate
        ring_size -= ring_width
        # flip the wood type for the next ring
        use_early_wood = ~use_early_wood

    if final_resize_height < final_stretch_height or final_resize_width < final_stretch_width:
        # stretch the image in the height of the rings are at a vertical angle
        image = util.img_as_ubyte(transform.resize(image, (final_stretch_height, final_stretch_width)))

    if gaussian_blur > 0:
        # add gaussian blus to the wood rings
        image = gaussian_filter(image, sigma=gaussian_blur)

    # create the 3d array by moving linearly over the strechted image with the original height window
    image3d = np.array([image[
        int((final_stretch_height-stretch_height)/2) + int(i*(stretch_height-height)/depth):
        int((final_stretch_height-stretch_height)/2) + int(height + i*(stretch_height-height)/depth),
        int((final_stretch_width-stretch_width)/2) + int(i*(stretch_width-width)/depth):
        int((final_stretch_width-stretch_width)/2) + int(width + i*(stretch_width-width)/depth)] for i in range(depth)])

    return image3d


def interactive_slice_viewer(image3d):
    """
    Displays an interactive slider to scroll through slices of a 3D greyscale array along different axes.

    Parameters:
    - image3d: 3D NumPy array (greyscale values).
    """
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.3)

    # Initial view settings
    current_axis = 0  # 0 = axial, 1 = coronal, 2 = sagittal
    slice_index = 0
    max_slices = [image3d.shape[0] - 1, image3d.shape[1] - 1, image3d.shape[2] - 1]
    dimensions = [(image3d.shape[1], image3d.shape[2]), (image3d.shape[2],
                                                         image3d.shape[0]), (image3d.shape[1], image3d.shape[0])]

    slider_ax = plt.axes([0.42, 0.1, 0.5, 0.05])

    radio_ax = plt.axes([0.01, 0.01, 0.25, 0.2])
    radio_buttons = RadioButtons(radio_ax, ('Front-to-Back', 'Top-to-Bottom', 'Left-to-Right'))

    slice_slider, img_display = None, None

    def update_slice(val):
        slice_idx = int(slice_slider.val)
        if current_axis == 0:
            img_display.set_data(image3d[slice_idx, :, :])
        elif current_axis == 1:
            img_display.set_data(image3d[:, slice_idx, :].T)
        elif current_axis == 2:
            img_display.set_data(image3d[:, :, slice_idx].T)

    def update_axis(label):
        nonlocal current_axis
        nonlocal img_display
        nonlocal slice_slider

        if label == 'Front-to-Back':
            current_axis = 0
        elif label == 'Top-to-Bottom':
            current_axis = 1
        elif label == 'Left-to-Right':
            current_axis = 2

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
        img_display = ax.imshow(image3d[0, :, :] if current_axis == 0 else image3d[:, 0, :].T
                                if current_axis == 1 else image3d[:, :, 0].T, cmap='gray')
        ax.set_xlim([0, dimensions[current_axis][1]])
        ax.set_ylim([dimensions[current_axis][0], 0])

    radio_buttons.on_clicked(update_axis)
    update_axis('Front-to-Back')
    plt.show()


# Example usage:
if __name__ == "__main__":
    image3d = generate_wood_block(seed=1, rot_deg_height=0, rot_deg_width=0)
    interactive_slice_viewer(image3d)
