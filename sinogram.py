from phantom import generate_wood_block
import numpy as np
import astra
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import imageio
import os


def interactive_sinogram_viewer(image3d):
    """
    Displays an interactive slider to scroll through slices of a 3D sinogram.

    Parameters:
    - image3d: 3D NumPy array (greyscale values).
    """
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.3)

    slice_index = 0
    max_slices = image3d.shape[1] - 1

    slider_ax = plt.axes([0.3, 0.1, 0.5, 0.05])

    def update_slice(val):
        slice_idx = int(slice_slider.val)
        img_display.set_data(image3d[:, slice_idx, :])

    slice_slider = Slider(
        ax=slider_ax,
        label='Projection Index',
        valmin=0,
        valmax=max_slices,
        valinit=slice_index,
        valstep=int(max_slices/20)+1
    )
    slice_slider.on_changed(update_slice)

    img_display = ax.imshow(image3d[:, 0, :], cmap='gray')

    plt.show()


# Create phantom
phantom = generate_wood_block(0)
phantom_depth, phantom_height, phantom_width = phantom.shape

# Create geometries and projector for linear trajectory
# z is phantom depth
# y is phantom height
# x is phantom width (linear trajectory axis)
# create_vol_geom uses order Y, X, Z
phantom_depth_sample_interval = 200
phantom_reduced = phantom[::phantom_depth_sample_interval]
num_depth_slices = phantom_reduced.shape[0]
vol_geom = astra.create_vol_geom(phantom_height, phantom_width, num_depth_slices, 0,
                                 phantom_width, 0, phantom_height, 0, phantom_depth)
num_projections = 21
vectors = np.zeros((num_projections, 12))

cone_angle = 9
cone_angle = np.deg2rad(cone_angle)

srcZ = (((phantom_width / 2) / np.tan(cone_angle))) * -1
srcY = phantom_height / 2
srcX_start = ((-1 * srcZ + phantom_depth) * np.tan(cone_angle)) + phantom_width
srcX_end = -1 * ((-1 * srcZ + phantom_depth) * np.tan(cone_angle))
srcX_shift = -1 * (np.abs(srcX_start) + np.abs(srcX_end)) / (num_projections - 1)

det_row_count = phantom_width
det_col_count = phantom_width

det_pix_scale = 2

uZ, uY, uX = (0, 0, det_pix_scale)
vZ, vY, vX = (0, det_pix_scale, 0)

dZ = ((det_pix_scale * phantom_width / 2) / np.tan(cone_angle)) + srcZ
dY = srcY
dX_start = srcX_start
dX_end = srcX_end
dX_shift = srcX_shift


for i in range(num_projections):
    srcX = srcX_start + i * srcX_shift
    dX = dX_start + i * dX_shift
    vectors[i, :] = [srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ]
proj_geom = astra.create_proj_geom('cone_vec', det_row_count, det_col_count, vectors)

id_phantom = astra.data3d.create('-vol', vol_geom, phantom_reduced)
id_sinogram, sinogram = astra.create_sino3d_gpu(id_phantom, proj_geom, vol_geom, gpuIndex=0)

sinogram_normalized = sinogram * 255.0/sinogram.max()

os.makedirs("sinograms", exist_ok=True)
for proj in range(num_projections):
    imageio.imwrite(f"sinograms/sino_{proj}.png", sinogram_normalized[:, proj, :].astype(np.uint8))
