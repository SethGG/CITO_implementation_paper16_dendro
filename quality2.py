import numpy as np
from scipy.ndimage import map_coordinates
from phantom import generate_wood_block
from sinogram import create_reconstruction
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from skimage.metrics import structural_similarity as ssim
import os
from tqdm import tqdm
import imageio.v2 as imageio


# Helper functions for radius and angle calculations


def compute_radius_for_angle(image, center, theta):
    height, width = image.shape
    x0, y0 = center
    dx, dy = np.cos(theta), np.sin(theta)

    t_vals = []

    # Intersect with vertical boundaries: x = 0 and x = width-1
    if dx != 0:
        for x_bound in [0, width - 1]:
            t = (x_bound - x0) / dx
            if t <= 0:
                continue
            y = y0 + t * dy
            if 0 <= y < height:
                t_vals.append(t)

    # Intersect with horizontal boundaries: y = 0 and y = height-1
    if dy != 0:
        for y_bound in [0, height - 1]:
            t = (y_bound - y0) / dy
            if t <= 0:
                continue
            x = x0 + t * dx
            if 0 <= x < width:
                t_vals.append(t)

    if not t_vals:
        return 0

    return max(t_vals)


# Step 1: Extract radial profile


def extract_radial_profile(image, center, theta, radius, pixel_step=0.5):
    x0, y0 = center
    radii = np.arange(0, radius, pixel_step)
    x_coords = x0 + radii * np.cos(theta)
    y_coords = y0 + radii * np.sin(theta)

    height, width = image.shape
    mask = (x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < height)

    profile = map_coordinates(image, [y_coords[mask], x_coords[mask]], order=1)
    return profile

# Step 2: Derivative of radial profile to find ring boundaries


def find_ring_boundaries(profile, smoothing_window=20, prominence=0.1):
    if len(profile) < smoothing_window:
        return [], [], []
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


def plot_radial_correlation(angles, all_r, all_r_detrend, all_num_rings, limited_indexes, image, image_recon):
    fig = plt.figure(figsize=(12, 10))
    ax0 = fig.add_subplot(221)
    ax3 = fig.add_subplot(222)
    ax1 = fig.add_subplot(234, polar=True)
    ax4 = fig.add_subplot(235, polar=True)
    ax2 = fig.add_subplot(236, polar=True)

    ax1.plot(angles, all_r, color='blue', label='Correlation (r)')
    all_r_upper = all_r.copy()
    all_r_upper[~limited_indexes] = 0
    ax1.fill(angles, all_r_upper, color='skyblue', alpha=0.3)
    all_r_lower = all_r.copy()
    all_r_lower[limited_indexes] = 0
    ax1.fill(angles, all_r_lower, color='lightcoral', alpha=0.3)
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


def create_phantom_and_reconstruction(seed, rot_deg_height, rot_deg_width, depth_sample_interval,
                                      cone_angle_deg, output_dir="output"):
    # Define directories
    phantom_dir = os.path.join(output_dir, "phantoms")
    recon_dir = os.path.join(output_dir, "reconstructions")
    os.makedirs(phantom_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)

    # Define base filename and slice subdirs
    filename = f"seed_{seed}_roth_{rot_deg_height}_rotw_{rot_deg_width}_dsi_{depth_sample_interval}_cangle_{cone_angle_deg}"
    phantom_path = os.path.join(phantom_dir, f"{filename}.npy")
    center_path = os.path.join(phantom_dir, f"center_{filename}.npy")
    recon_path = os.path.join(recon_dir, f"{filename}.npy")
    recon_img_dir = os.path.join(recon_dir, filename)

    # Load or generate phantom and center data
    if os.path.exists(phantom_path) and os.path.exists(center_path):
        image3d = np.load(phantom_path)
        slices_center = np.load(center_path)
    else:
        full_image3d, full_slices_center = generate_wood_block(
            seed=seed, rot_deg_height=rot_deg_height, rot_deg_width=rot_deg_width, center_pith=True)
        # Subsample slices
        image3d = full_image3d[::depth_sample_interval]
        slices_center = full_slices_center[::depth_sample_interval]

        np.save(phantom_path, image3d)
        np.save(center_path, slices_center)

    # Load or generate reconstructions
    if os.path.exists(recon_path):
        recon3d = np.load(recon_path)
    else:
        _, full_recon3d = create_reconstruction(full_image3d, depth_sample_interval=depth_sample_interval,
                                                cone_angle_deg=cone_angle_deg)
        # Convert to uint8
        recon3d = np.clip(full_recon3d, 0, 255).astype(np.uint8)
        np.save(recon_path, recon3d)

        # Save recon slices as images
        os.makedirs(recon_img_dir, exist_ok=True)
        for i, img in enumerate(recon3d):
            imageio.imwrite(os.path.join(recon_img_dir, f"recon_slice_{i:03d}.png"), img)

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


def find_best_reconstruction_depth(phantom_recon):
    image3d, slices_center, recon3d = phantom_recon

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
            if len(bl_profile) == 0:
                all_ring_widths.append([])
                all_r.append(np.nan)
                continue

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
        slices_ring_weighted_r.append(np.sum(all_ring_weigthed_r[~np.isnan(all_ring_weigthed_r)]))
        slices_average_r.append(np.mean(all_r[~np.isnan(all_r)]))

        # Only for angles with r above the 50th percentile
        # threshold_r = np.percentile(all_r[~np.isnan(all_r)], 60)
        # limited_indexes = all_r > threshold_r

        # Only top 10 percentile ring numbers
        threshold_num_rings = np.percentile(all_num_rings, 80)
        limited_indexes = all_num_rings >= threshold_num_rings
        all_num_rings_limited = all_num_rings[limited_indexes]
        all_r_limited = all_r[limited_indexes]

        all_ring_weigthed_r_limited = ring_weighted_correlations(all_r_limited, all_num_rings_limited)
        slices_ring_weighted_r_limited.append(
            np.sum(all_ring_weigthed_r_limited[~np.isnan(all_ring_weigthed_r_limited)]))
        slices_average_r_limited.append(np.mean(all_r_limited[~np.isnan(all_r_limited)]))

        # plot_radial_correlation(angles, all_r, all_ring_weigthed_r, all_num_rings, limited_indexes,
        #                        phantom_slice, recon_slice)

    plot_reconstruction_quality_summary(image3d, recon3d, slices_ring_weighted_r, slices_average_r,
                                        slices_ring_weighted_r_limited, slices_average_r_limited)


# Experiments
if __name__ == "__main__":
    # create phantoms and reconstructions
    phantom_recon_1 = create_phantom_and_reconstruction(
        seed=4, rot_deg_height=0, rot_deg_width=0, depth_sample_interval=20, cone_angle_deg=9)
    phantom_recon_2 = create_phantom_and_reconstruction(
        seed=4, rot_deg_height=8, rot_deg_width=0, depth_sample_interval=20, cone_angle_deg=9)
    phantom_recon_3 = create_phantom_and_reconstruction(
        seed=4, rot_deg_height=0, rot_deg_width=8, depth_sample_interval=20, cone_angle_deg=9)
    phantom_recon_4 = create_phantom_and_reconstruction(
        seed=4, rot_deg_height=0, rot_deg_width=10, depth_sample_interval=20, cone_angle_deg=9)
    phantom_recon_5 = create_phantom_and_reconstruction(
        seed=4, rot_deg_height=0, rot_deg_width=12, depth_sample_interval=20, cone_angle_deg=9)

    # find_best_reconstruction_depth(phantom_recon_1)
    # find_best_reconstruction_depth(phantom_recon_2)
    # find_best_reconstruction_depth(phantom_recon_3)
    # find_best_reconstruction_depth(phantom_recon_4)
    find_best_reconstruction_depth(phantom_recon_5)
