from phantom import generate_wood_block, interactive_slice_viewer
import numpy as np
import astra

# Create phantom
phantom = generate_wood_block(0)
phantom_depth, phantom_height, phantom_width = phantom.shape

# Create geometries and projector for linear trajectory
# z is phantom depth
# y is phantom height
# x is phantom width (linear trajectory axis)
# create_vol_geom uses order Y, X, Z
vol_geom = astra.create_vol_geom((phantom_height, phantom_width, phantom_depth))
num_projections = 2
vectors = np.zeros((num_projections, 12))

cone_angle = 9
cone_angle = np.deg2rad(cone_angle)

srcZ = (((phantom_width / 2) / np.tan(cone_angle)) + (phantom_depth / 2)) * -1
srcY = 0
srcX_start = (srcZ - (phantom_depth / 2)) * np.tan(cone_angle)
srcX_end = srcX_start * -1
srcX_shift = 2 * srcX_end / (num_projections - 1)

det_row_count = phantom_width
det_col_count = phantom_width

det_pix_scale = 2

uZ, uY, uX = (0, 0, det_pix_scale)
vZ, vY, vX = (0, det_pix_scale, 0)

dZ = ((det_pix_scale * phantom_width / 2) / np.tan(cone_angle)) + srcZ
dY = 0
dX_start = srcX_start
dX_end = srcX_end
dX_shift = srcX_shift


for i in range(num_projections):
    srcX = srcX_start + i * srcX_shift
    dX = dX_start + i * dX_shift
    vectors[i, :] = [srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ]
proj_geom = astra.create_proj_geom('cone_vec', det_row_count, det_col_count, vectors)

astra.set_gpu_index(0)
id, volume = astra.create_sino3d_gpu(phantom, proj_geom, vol_geom, gpuIndex=0)

breakpoint()
