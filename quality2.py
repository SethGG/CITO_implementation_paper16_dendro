import numpy as np
from scipy.ndimage import map_coordinates
from phantom import generate_wood_block, interactive_slice_viewer
from sinogram import create_reconstruction
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
import os
from tqdm import tqdm


# Helper functions for radius and angle calculations


def compute_radius_for_angle(image, center, theta):
    height, width = image.shape
    x0, y0 = center
    dx, dy = np.cos(theta), np.sin(theta)

    bounds = []

    if dx != 0:
        t1 = (0 - x0) / dx
        t2 = (width - 1 - x0) / dx
        bounds.extend([t for t in (t1, t2) if t > 0])

    if dy != 0:
        t3 = (0 - y0) / dy
        t4 = (height - 1 - y0) / dy
        bounds.extend([t for t in (t3, t4) if t > 0])

    if not bounds:
        return 0  # direction is completely away from image

    return min(bounds)


def compute_angle_for_max_radius(image, center):
    height, width = image.shape
    x0, y0 = center

    # Define the four corners of the image
    corners = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]

    max_dist_sq = 0
    best_theta = 0

    for x, y in corners:
        dx = x - x0
        dy = y - y0
        dist_sq = dx**2 + dy**2
        if dist_sq > max_dist_sq:
            max_dist_sq = dist_sq
            best_theta = np.arctan2(dy, dx)

    return best_theta


# Step 1: Extract radial profile


def extract_radial_profile(image, center, theta, radius, pixel_step=0.5):
    x0, y0 = center
    radii = np.arange(0, radius, pixel_step)
    x_coords = x0 + radii * np.cos(theta)
    y_coords = y0 + radii * np.sin(theta)
    profile = map_coordinates(image, [y_coords, x_coords], order=1)
    return profile

# Step 2: Derivative of radial profile to find ring boundaries


def find_ring_boundaries(profile, smoothing_window=20, prominence=0.4):
    derivative = savgol_filter(profile, window_length=smoothing_window, polyorder=1, deriv=1)
    peaks_max, _ = find_peaks(derivative, prominence=prominence*np.mean(np.abs(derivative)), height=0)
    peaks_min, _ = find_peaks(-derivative, prominence=prominence*np.mean(np.abs(derivative)), height=0)
    peaks = np.concatenate((peaks_max, peaks_min))
    peaks.sort()
    ring_widths = np.diff(peaks)
    return derivative, peaks, ring_widths


# Step 3: Standardize series using Baillie-Pilcher method


def bp73_standardize(widths):
    standardized = []
    for i in range(2, len(widths) - 2):
        local_mean = np.mean(widths[i-2:i+3])
        if local_mean == 0:
            standardized.append(0)
        else:
            val = np.log((100 * widths[i]) / local_mean)
            # print(f"ring: {i} local mean: {local_mean} val: {val}")
            standardized.append(val)
    return np.array(standardized)


# Step 4: Compute r and GL

def fast_corr(series1, series2):
    min_len = min(len(series1), len(series2))
    x = np.asarray(series1[:min_len], dtype=float).copy()
    y = np.asarray(series2[:min_len], dtype=float).copy()
    x -= x.mean()
    y -= y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return np.dot(x, y) / denom if denom != 0 else 0

# Helper functions to compare ring detection


def plot_peak_detection(derivative_1, peaks_1, ring_widths_1, std_ring_widths_1,
                        derivative_2, peaks_2, ring_widths_2, std_ring_widths_2):
    fig = plt.figure(figsize=(12, 6))
    ax0 = fig.add_subplot(321)
    ax1 = fig.add_subplot(322)
    ax2 = fig.add_subplot(323)
    ax3 = fig.add_subplot(324)
    ax4 = fig.add_subplot(325)
    ax5 = fig.add_subplot(326)

    ax0.plot(derivative_1)
    ax0.plot(peaks_1, derivative_1[peaks_1], 'x')
    ax1.plot(derivative_2)
    ax1.plot(peaks_2, derivative_2[peaks_2], 'x')

    ax2.plot(ring_widths_1)
    ax3.plot(ring_widths_2)

    ax4.plot(std_ring_widths_1)
    ax5.plot(std_ring_widths_2)

    plt.show()

# Plot radial correlation and ring numbers in recontruction


def plot_radial_correlation(angles, all_r, all_r_detrend, all_num_rings, image, image_recon):
    fig = plt.figure(figsize=(12, 10))
    ax0 = fig.add_subplot(221)
    ax3 = fig.add_subplot(222)
    ax1 = fig.add_subplot(234, polar=True)
    ax4 = fig.add_subplot(235, polar=True)
    ax2 = fig.add_subplot(236, polar=True)

    ax1.plot(angles, all_r, color='blue', label='Correlation (r)')
    ax1.fill(angles, all_r, color='skyblue', alpha=0.3)
    ax1.set_theta_zero_location("E")  # 0° at the right
    ax1.set_theta_direction(-1)        # Counterclockwise
    # ax1.set_title("Radial correlation of ring widths to baseline profile")
    ax1.legend(loc='upper right')

    ax4.plot(angles, all_r_detrend, color='blue', label='Correlation Detrended (r)')
    ax4.fill(angles, all_r_detrend, color='skyblue', alpha=0.3)
    ax4.set_theta_zero_location("E")  # 0° at the right
    ax4.set_theta_direction(-1)        # Counterclockwise
    # ax1.set_title("Radial correlation of ring widths to baseline profile")
    ax4.legend(loc='upper right')

    ax2.plot(angles, all_num_rings, color='orange', label='Number of rings')
    ax2.fill(angles, all_num_rings, color='moccasin', alpha=0.3)
    ax2.set_theta_zero_location("E")  # 0° at the right
    ax2.set_theta_direction(-1)        # Counterclockwise
    # ax2.set_title("Radial number of tree-rings")
    ax2.legend(loc='upper right')

    ax0.imshow(image, cmap='gray')
    ax0.set_title("Phantom ground truth")

    ax3.imshow(image_recon, cmap='gray')
    ax3.set_title("Reconstruction")

    plt.show()


def create_phantom_and_reconstruction(seed, rot_deg_height, rot_deg_width, depth_sample_interval, cone_angle_deg):
    filename = \
        f"seed_{seed}_roth_{rot_deg_height}_rotw_{rot_deg_width}_dsi_{depth_sample_interval}_cangle_{cone_angle_deg}.npy"

    if os.path.exists(f"phantom_{filename}") and os.path.exists(f"center_{filename}"):
        image3d = np.load(f"phantom_{filename}")
        slices_center = np.load(f"center_{filename}")
    else:
        image3d, slices_center = generate_wood_block(
            seed=seed, rot_deg_height=rot_deg_height, rot_deg_width=rot_deg_width)
        np.save(f"phantom_{filename}", image3d)
        np.save(f"center_{filename}", slices_center)

    if os.path.exists(f"recon_{filename}"):
        recon3d = np.load(f"recon_{filename}")
    else:
        _, recon3d = create_reconstruction(image3d, depth_sample_interval=depth_sample_interval,
                                           cone_angle_deg=cone_angle_deg)
        np.save(f"recon_{filename}", recon3d)

    # get only the slices in the phantom which where reconstructed
    image3d = image3d[::depth_sample_interval]
    slices_center = slices_center[::depth_sample_interval]

    return image3d, slices_center, recon3d


def plot_reconstruction_quality_summary(image3d, recon3d,
                                        slices_ring_weighted_r,
                                        slices_average_r,
                                        slices_ring_weighted_r_limited,
                                        slices_average_r_limited,
                                        title="Reconstruction Quality Across Depth"):
    """
    Plots ring-weighted correlation, average correlation, and SSIM across slices.

    Parameters:
    - image3d: phantom 3D array
    - recon3d: reconstructed 3D array
    - slices_ring_weighted_r: list of weighted correlation scores per slice
    - slices_average_r: list of average correlation scores per slice
    - title: plot title
    """
    num_slices = len(image3d)
    depths = np.arange(num_slices)

    # Compute SSIM per slice
    ssim_values = []
    for phantom_slice, recon_slice in zip(image3d, recon3d):
        score = ssim(phantom_slice, recon_slice, data_range=phantom_slice.max() - phantom_slice.min())
        ssim_values.append(score)

    # Prepare plot
    plt.figure(figsize=(8, 6))

    # Plot each line and its max point
    def plot_line_with_max(y_values, label, color):
        plt.plot(depths, y_values, label=label, color=color)
        max_idx = np.argmax(y_values)
        max_val = y_values[max_idx]
        plt.scatter(depths[max_idx], max_val, color=color, edgecolor='black', zorder=5)
        plt.text(depths[max_idx], max_val - 0.01,
                 f'slice {max_idx}\n{max_val:.3f}',
                 color=color, ha='center', va='bottom', fontsize=10)

    plot_line_with_max(slices_ring_weighted_r, "Ring-weighted r", 'blue')
    plot_line_with_max(slices_ring_weighted_r_limited, "Ring-weighted r (Horizontal angles only)", 'orange')
    plot_line_with_max(ssim_values, "SSIM", 'green')

    plt.xlabel("Slice index (depth)")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def ring_weighted_correlations(correlations, num_rings):
    """
    Option 1: Simple ring-weighted average correlation.
    """
    correlations = np.asarray(correlations)
    num_rings = np.asarray(num_rings)
    weights = num_rings / num_rings.sum()

    return weights * correlations


def find_best_reconstruction_depth(seed):
    image3d, slices_center, recon3d = create_phantom_and_reconstruction(
        seed=seed, rot_deg_height=0, rot_deg_width=0, depth_sample_interval=20, cone_angle_deg=9)

    angles = np.deg2rad(np.arange(360))

    slices_average_r = []
    slices_ring_weighted_r = []
    slices_average_r_limited = []
    slices_ring_weighted_r_limited = []

    for phantom_slice, slice_center, recon_slice in tqdm(zip(image3d, slices_center, recon3d), total=len(image3d)):

        all_ring_widths = []
        all_r = []

        for theta in angles:
            bl_radius = compute_radius_for_angle(phantom_slice, slice_center, theta)
            bl_profile = extract_radial_profile(phantom_slice, slice_center, theta, bl_radius)
            _, _, bl_ring_widths = find_ring_boundaries(bl_profile)
            all_ring_widths.append(bl_ring_widths)

            radius = compute_radius_for_angle(recon_slice, slice_center, theta)
            profile = extract_radial_profile(recon_slice, slice_center, theta, radius)
            r = fast_corr(bl_profile, profile)
            all_r.append(r)

        all_num_rings = np.array([len(x) for x in all_ring_widths])
        all_r = np.array(all_r)

        # For all angles
        all_ring_weigthed_r = ring_weighted_correlations(all_r, all_num_rings)
        slices_ring_weighted_r.append(np.sum(all_ring_weigthed_r))
        slices_average_r.append(np.mean(all_r))

        # Only for angles 0 to 45, 135 to 225, and 315 to 0
        indices = np.concatenate([
            np.arange(0, 46),
            np.arange(135, 226),
            np.arange(315, 360)
        ])
        all_num_rings_limited = all_num_rings[indices]
        all_r_limited = all_r[indices]

        all_ring_weigthed_r_limited = ring_weighted_correlations(all_r_limited, all_num_rings_limited)
        slices_ring_weighted_r_limited.append(np.sum(all_ring_weigthed_r_limited))
        slices_average_r_limited.append(np.mean(all_r_limited))

        # plot_radial_correlation(angles, all_r, all_ring_weigthed_r, all_num_rings,
        #                        phantom_slice, recon_slice)

    plot_reconstruction_quality_summary(image3d, recon3d, slices_ring_weighted_r, slices_average_r,
                                        slices_ring_weighted_r_limited, slices_average_r_limited)


def test_on_gaussian():
    image3d, slices_center = generate_wood_block(seed=4, rot_deg_height=45, rot_deg_width=20)

    slice_num = 50
    image = image3d[slice_num]
    center = slices_center[slice_num]

    # find the angle of and radius of the longest line from the center (baseline)
    bl_theta = compute_angle_for_max_radius(image, center)
    bl_radius = compute_radius_for_angle(image, center, bl_theta)
    # extract radial baseline radial profile
    bl_profile = extract_radial_profile(image, center, bl_theta, bl_radius)
    # extract the ring edge locations anc calculate ring widths
    bl_derivative, bl_peaks, bl_ring_widths = find_ring_boundaries(bl_profile)
    # standardize the ring width series
    bl_standard_ring_widths = bp73_standardize(bl_ring_widths)

    # Test on a noisy image as reconstruction placeholder
    image_recon = gaussian_filter(image, sigma=5)

    all_profiles = []
    all_derivatives = []
    all_peaks = []
    all_ring_widths = []
    all_standard_ring_widths = []

    all_r = []
    all_gl = []

    angles = np.deg2rad(np.arange(360)) + bl_theta

    for theta in angles:
        radius = compute_radius_for_angle(image_recon, center, theta)
        profile = extract_radial_profile(image_recon, center, theta, radius)
        all_profiles.append(profile)
        derivative, peaks, ring_widths = find_ring_boundaries(profile)
        all_derivatives.append(derivative)
        all_peaks.append(peaks)
        all_ring_widths.append(ring_widths)
        standard_ring_widths = bp73_standardize(ring_widths)
        all_standard_ring_widths.append(standard_ring_widths)
        tbp, r = compute_tbp_bp73(bl_standard_ring_widths, standard_ring_widths)
        all_r.append(r)
        gl = compute_gl(bl_ring_widths, ring_widths)
        all_gl.append(gl)

    # all_num_rings = [len(x) for x in all_ring_widths]

    # plot_radial_correlation(angles, bl_theta, bl_radius, all_r, all_gl, all_num_rings, image, center, image_recon)

    score, segments = dp_segmented_correlation_shifted(
        reference=bl_standard_ring_widths,
        recon_segments=all_standard_ring_widths,
        min_segment_len=20,
        max_segment_len=40,
        shift_max=3,
        n_jobs=-1  # use all available CPU cores
    )

    print(f"DP Optimal Weighted Correlation Score: {score:.3f}")
    for start, end, angle_idx, r in segments:
        print(f"Rings {start}-{end} → angle {angle_idx} → r = {r:.3f}")

    plot_segment_marks_on_image(
        image=image_recon,
        center=center,
        segments=segments,
        angles=angles,
        ring_boundaries_all=all_peaks,
        pixel_step=0.5
    )


# Example usage:
if __name__ == "__main__":
    # test_on_gaussian()
    find_best_reconstruction_depth(seed=4)
