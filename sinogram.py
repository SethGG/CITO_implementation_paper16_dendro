"""
CITO 24-25
Daniël Zee (s2063131) and Martijn Combé (s2599406)
"""

from phantom import generate_wood_block
import numpy as np
import astra
import imageio
import os


def create_reconstruction(phantom, depth_sample_interval=20, num_projections=101, cone_angle_deg=9, iterations=50):
    """
    Creates sinogram and reconstructs a 3D volume using ASTRA cone-beam geometry.

    Parameters:
        phantom: 3D numpy array (z = depth, y = height, x = width)
        depth_sample_interval: step size to downsample depth
        num_projections: number of angular projections
        cone_angle_deg: cone angle in degrees
        iterations: number of SIRT iterations

    Returns:
        sinogram: 3D array of simulated projections
        recon: 3D reconstructed volume (uint8)
    """
    # Phantom shape convention: z = depth (slice), y = height, x = width
    phantom_depth, phantom_height, phantom_width = phantom.shape

    # Downsample depth
    phantom_reduced = phantom[::depth_sample_interval]
    num_depth_slices = phantom_reduced.shape[0]

    # Create volume geometry for ASTRA (y, x, z order)
    vol_geom = astra.create_vol_geom(phantom_height, phantom_width, num_depth_slices,
                                     0, phantom_width, 0, phantom_height, 0, phantom_depth)

    # Define cone-beam projection vectors for a linear trajectory
    vectors = np.zeros((num_projections, 12))
    cone_angle = np.deg2rad(cone_angle_deg)

    # Compute source and detector geometry based on cone angle
    srcZ = (((phantom_width / 2) / np.tan(cone_angle))) * -1
    srcY = phantom_height / 2
    srcX_start = ((-1 * srcZ + phantom_depth) * np.tan(cone_angle)) + phantom_width
    srcX_end = -1 * ((-1 * srcZ + phantom_depth) * np.tan(cone_angle))
    srcX_shift = -1 * (np.abs(srcX_start) + np.abs(srcX_end)) / (num_projections - 1)

    det_row_count = phantom_height
    det_col_count = phantom_width
    det_pix_scale = 3

    # Unit vectors u and v for detector plane orientation
    uZ, uY, uX = (0, 0, det_pix_scale)
    vZ, vY, vX = (0, det_pix_scale, 0)

    # Detector position (relative to source)
    dZ = ((det_pix_scale * phantom_width / 2) / np.tan(cone_angle)) + srcZ
    dY = srcY
    dX_start = srcX_start
    dX_shift = srcX_shift

    # Fill vector geometry
    for i in range(num_projections):
        srcX = srcX_start + i * srcX_shift
        dX = dX_start + i * dX_shift
        vectors[i, :] = [srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ]

    proj_geom = astra.create_proj_geom('cone_vec', det_row_count, det_col_count, vectors)

    # Create ASTRA phantom object and compute sinogram
    phantom_id = astra.data3d.create('-vol', vol_geom, phantom_reduced)
    sinogram_id, sinogram = astra.create_sino3d_gpu(phantom_id, proj_geom, vol_geom)

    # Create reconstruction volume and run SIRT3D
    recon_id = astra.data3d.create('-vol', vol_geom)
    cfg = astra.astra_dict('SIRT3D_CUDA')
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ReconstructionDataId'] = recon_id
    cfg['option'] = {'MinConstraint': 0}  # Enforce non-negativity constraint
    sirt_id = astra.algorithm.create(cfg)
    astra.algorithm.run(sirt_id, iterations)

    # Retrieve result and convert to uint8
    recon = astra.data3d.get(recon_id)
    recon = recon.astype(np.uint8)

    # Free GPU memory and cleanup
    astra.data3d.delete([phantom_id, sinogram_id, recon_id])
    astra.algorithm.delete(sirt_id)

    return sinogram, recon


if __name__ == "__main__":

    # Experiment loop over ring tilt and cone angle
    ring_tilts = [0, 8, 15]
    cone_angles = [9, 18]

    for ring_tilt in ring_tilts:
        for cone_angle_deg in cone_angles:

            # Generate synthetic wood block phantom
            phantom = generate_wood_block(0, rot_deg_width=ring_tilt)
            phantom_depth, phantom_height, phantom_width = phantom.shape

            # Downsample phantom in z-direction
            phantom_depth_sample_interval = 20
            phantom_reduced = phantom[::phantom_depth_sample_interval]
            num_depth_slices = phantom_reduced.shape[0]

            # Volume geometry
            vol_geom = astra.create_vol_geom(phantom_height, phantom_width, num_depth_slices,
                                             0, phantom_width, 0, phantom_height, 0, phantom_depth)

            num_projections = 101
            vectors = np.zeros((num_projections, 12))
            cone_angle = np.deg2rad(cone_angle_deg)

            # Define geometry for source and detector
            srcZ = (((phantom_width / 2) / np.tan(cone_angle))) * -1
            srcY = phantom_height / 2
            srcX_start = ((-1 * srcZ + phantom_depth) * np.tan(cone_angle)) + phantom_width
            srcX_end = -1 * ((-1 * srcZ + phantom_depth) * np.tan(cone_angle))
            srcX_shift = -1 * (np.abs(srcX_start) + np.abs(srcX_end)) / (num_projections - 1)

            det_row_count = phantom_height
            det_col_count = phantom_width
            det_pix_scale = 3

            uZ, uY, uX = (0, 0, det_pix_scale)
            vZ, vY, vX = (0, det_pix_scale, 0)

            dZ = ((det_pix_scale * phantom_width / 2) / np.tan(cone_angle)) + srcZ
            dY = srcY
            dX_start = srcX_start
            dX_shift = srcX_shift

            for i in range(num_projections):
                srcX = srcX_start + i * srcX_shift
                dX = dX_start + i * dX_shift
                vectors[i, :] = [srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ]

            proj_geom = astra.create_proj_geom('cone_vec', det_row_count, det_col_count, vectors)

            # Simulate sinogram
            phantom_id = astra.data3d.create('-vol', vol_geom, phantom_reduced)
            sinogram_id, sinogram = astra.create_sino3d_gpu(phantom_id, proj_geom, vol_geom)
            sinogram_normalized = sinogram * 255.0 / sinogram.max()

            # Save sinogram slices
            sinogram_dirname = f"sinograms/ring_tilt_{ring_tilt}_cone_angle_{cone_angle_deg}"
            os.makedirs(sinogram_dirname, exist_ok=True)
            for proj in range(num_projections):
                imageio.imwrite(f"{sinogram_dirname}/sino_{proj}.png", sinogram_normalized[:, proj, :].astype(np.uint8))

            # Perform reconstruction
            recon_id = astra.data3d.create('-vol', vol_geom)
            cfg = astra.astra_dict('SIRT3D_CUDA')
            cfg['ProjectionDataId'] = sinogram_id
            cfg['ReconstructionDataId'] = recon_id
            cfg['option'] = {'MinConstraint': 0}
            sirt_id = astra.algorithm.create(cfg)
            astra.algorithm.run(sirt_id, 50)
            recon = astra.data3d.get(recon_id)

            # Save reconstruction slices
            recon_dirname = f"reconstructions/ring_tilt_{ring_tilt}_cone_angle_{cone_angle_deg}"
            os.makedirs(recon_dirname, exist_ok=True)
            for slice in range(num_depth_slices):
                imageio.imwrite(f"{recon_dirname}/recon_{slice}.png", recon[slice, :, :].astype(np.uint8))

            # Cleanup ASTRA memory
            astra.data3d.delete([phantom_id, sinogram_id, recon_id])
            astra.algorithm.delete(sirt_id)
