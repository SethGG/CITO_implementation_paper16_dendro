"""
CITO 24-25
Daniël Zee (s2063131) and Martijn Combé (s2599406)
"""

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


# --- Ring Detection & Profile Extraction Functions ---

def compute_radius_for_angle(image, center, theta):
    """
    Computes the maximum radius along a given angle from the center until the image edge.
    """
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


def extract_radial_profile(image, center, theta, radius, pixel_step=0.5):
    """
    Extracts intensity values along a radial line from the center at a given angle.
    """
    x0, y0 = center
    radii = np.arange(0, radius, pixel_step)
    x_coords = x0 + radii * np.cos(theta)
    y_coords = y0 + radii * np.sin(theta)

    height, width = image.shape
    mask = (x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < height)

    profile = map_coordinates(image, [y_coords[mask], x_coords[mask]], order=1)
    return profile


def find_ring_boundaries(profile, smoothing_window=20, prominence=0.1):
    """
    Identifies ring boundaries in a radial profile using derivative peaks.
    """
    if len(profile) < smoothing_window:
        return [], [], []

    # First derivative of smoothed profile
    derivative = savgol_filter(profile, window_length=smoothing_window, polyorder=1, deriv=1)

    # Detect peaks in derivative (both up and down)
    peaks_max, _ = find_peaks(derivative, prominence=prominence*np.mean(np.abs(derivative)), height=0)
    peaks_min, _ = find_peaks(-derivative, prominence=prominence*np.mean(np.abs(derivative)), height=0)

    peaks = np.concatenate((peaks_max, peaks_min))
    peaks.sort()

    # Estimate ring widths as distances between peaks
    ring_widths = np.diff(peaks)
    return derivative, peaks, ring_widths


def fast_corr(series1, series2):
    """
    Computes normalized cross-correlation between two series.
    """
    min_len = min(len(series1), len(series2))
    x = np.asarray(series1[:min_len], dtype=float).copy()
    y = np.asarray(series2[:min_len], dtype=float).copy()
    x -= x.mean()
    y -= y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return np.dot(x, y) / denom if denom != 0 else 0


# --- Visualization Functions ---

def plot_peak_detection(derivative_1, peaks_1, ring_widths_1, std_ring_widths_1,
                        derivative_2, peaks_2, ring_widths_2, std_ring_widths_2):
    """
    Compares peak detection and standardization between two profiles.
    """
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


def plot_radial_correlation(angles, all_r, all_r_detrend, all_num_rings, limited_indexes, image, image_recon):
    """
    Plots phantom vs reconstruction and polar plots of ring metrics.
    """
    fig = plt.figure(figsize=(12, 10))
    ax0 = fig.add_subplot(221)
    ax3 = fig.add_subplot(222)
    ax1 = fig.add_subplot(234, polar=True)
    ax4 = fig.add_subplot(236, polar=True)
    ax2 = fig.add_subplot(235, polar=True)

    ax1.plot(angles, all_r, color='blue', label='Radial Correlation')
    all_r_upper = all_r.copy()
    all_r_upper[~limited_indexes] = 0
    ax1.fill(angles, all_r_upper, color='skyblue', alpha=0.3)
    all_r_lower = all_r.copy()
    all_r_lower[limited_indexes] = 0
    ax1.fill(angles, all_r_lower, color='lightcoral', alpha=0.3)
    ax1.set_theta_zero_location("E")
    ax1.set_theta_direction(-1)
    ax1.legend(loc='upper right')

    ax4.plot(angles, all_r_detrend, color='blue', label='Normalized Radial Correlation')
    ax4.fill(angles, all_r_detrend, color='skyblue', alpha=0.3)
    ax4.set_theta_zero_location("E")
    ax4.set_theta_direction(-1)
    ax4.legend(loc='upper right')

    ax2.plot(angles, all_num_rings, color='orange', label='Number of Rings')
    ax2.fill(angles, all_num_rings, color='moccasin', alpha=0.3)
    ax2.set_theta_zero_location("E")
    ax2.set_theta_direction(-1)
    ax2.legend(loc='upper right')

    ax0.imshow(image, cmap='gray')
    ax0.set_title("Phantom Ground Truth")

    ax3.imshow(image_recon, cmap='gray')
    ax3.set_title("Reconstruction")

    plt.show()


# --- Experiment Execution & Evaluation ---

def create_phantom_and_reconstruction(seed, rot_deg_height, rot_deg_width, depth_sample_interval,
                                      cone_angle_deg, output_dir="output"):
    """
    Loads or generates a phantom and its corresponding reconstruction.

    Saves output files as .npy and PNG slices.
    """
    phantom_dir = os.path.join(output_dir, "phantoms")
    recon_dir = os.path.join(output_dir, "reconstructions")
    os.makedirs(phantom_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)

    filename = f"seed_{seed}_roth_{rot_deg_height}_rotw_{rot_deg_width}_dsi_{depth_sample_interval}_cangle_{cone_angle_deg}"
    phantom_path = os.path.join(phantom_dir, f"{filename}.npy")
    center_path = os.path.join(phantom_dir, f"center_{filename}.npy")
    recon_path = os.path.join(recon_dir, f"{filename}.npy")
    recon_img_dir = os.path.join(recon_dir, filename)

    # Load or generate phantom
    if os.path.exists(phantom_path) and os.path.exists(center_path):
        image3d = np.load(phantom_path)
        slices_center = np.load(center_path)
    else:
        full_image3d, full_slices_center = generate_wood_block(
            seed=seed, rot_deg_height=rot_deg_height, rot_deg_width=rot_deg_width, center_pith=True)
        image3d = full_image3d[::depth_sample_interval].copy()
        slices_center = full_slices_center[::depth_sample_interval].copy()
        np.save(phantom_path, image3d)
        np.save(center_path, slices_center)

    # Load or generate reconstruction
    if os.path.exists(recon_path):
        recon3d = np.load(recon_path)
    else:
        _, full_recon3d = create_reconstruction(full_image3d, depth_sample_interval=depth_sample_interval,
                                                cone_angle_deg=cone_angle_deg)
        recon3d = np.clip(full_recon3d, 0, 255).astype(np.uint8)
        np.save(recon_path, recon3d)

        os.makedirs(recon_img_dir, exist_ok=True)
        for i, img in enumerate(recon3d):
            imageio.imwrite(os.path.join(recon_img_dir, f"recon_slice_{i:03d}.png"), img)

        del full_slices_center
        del full_image3d

    return image3d, slices_center, recon3d


def plot_reconstruction_quality_summary(results, labels, title, output_dir="output", suffix="", ylim=(0, 1)):
    """
    Plots per-slice metric scores (e.g., correlation, SSIM) across depth.
    """
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.ylim(ylim)

    def plot_line_with_max(y_values, label):
        depths = np.arange(len(y_values))
        plt.plot(depths, y_values, label=label)

    for result, label in zip(results, labels):
        plot_line_with_max(result, label)

    plt.xlabel("Slice Index (Depth)")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{title}{suffix}.png"))
    plt.show()


def ring_weighted_correlations(correlations, num_rings):
    """
    Computes ring-weighted average correlation per angle.
    """
    correlations = np.asarray(correlations)
    num_rings = np.asarray(num_rings)
    weights = num_rings / num_rings.sum()
    return weights * correlations


def find_best_reconstruction_depth(phantom_recon):
    """
    Evaluates a phantom-reconstruction pair across slices using ring-based metrics and SSIM.
    """
    image3d, slices_center, recon3d = phantom_recon
    angles = np.deg2rad(np.arange(360))

    slices_average_r = []
    slices_ring_weighted_r = []
    slices_average_r_limited = []
    slices_ring_weighted_r_limited = []
    slices_ssim_values = []

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

        # All angles
        all_ring_weigthed_r = ring_weighted_correlations(all_r, all_num_rings)
        slices_ring_weighted_r.append(np.sum(all_ring_weigthed_r[~np.isnan(all_ring_weigthed_r)]))
        slices_average_r.append(np.mean(all_r[~np.isnan(all_r)]))

        # Top 20% by ring count
        threshold_num_rings = np.percentile(np.unique(all_num_rings), 80)
        limited_indexes = all_num_rings >= threshold_num_rings
        all_num_rings_limited = all_num_rings[limited_indexes]
        all_r_limited = all_r[limited_indexes]
        all_ring_weigthed_r_limited = ring_weighted_correlations(all_r_limited, all_num_rings_limited)
        slices_ring_weighted_r_limited.append(
            np.sum(all_ring_weigthed_r_limited[~np.isnan(all_ring_weigthed_r_limited)]))
        slices_average_r_limited.append(np.mean(all_r_limited[~np.isnan(all_r_limited)]))

        # SSIM
        score = ssim(phantom_slice, recon_slice, data_range=phantom_slice.max() - phantom_slice.min())
        slices_ssim_values.append(score)

    return slices_ring_weighted_r, slices_ring_weighted_r_limited, slices_ssim_values


# --- Main Experiment Execution ---

if __name__ == "__main__":
    # Generate phantom and reconstructions for all experimental configurations

    # Baseline comparisons (no tilt)
    phantom_recon_1 = create_phantom_and_reconstruction(
        seed=4, rot_deg_height=0, rot_deg_width=0, depth_sample_interval=20, cone_angle_deg=9)
    phantom_recon_2 = create_phantom_and_reconstruction(
        seed=4, rot_deg_height=0, rot_deg_width=0, depth_sample_interval=20, cone_angle_deg=15)

    # Vary ring tilt (width) with cone angle 9°
    phantom_recon_3 = create_phantom_and_reconstruction(
        seed=4, rot_deg_height=0, rot_deg_width=6, depth_sample_interval=20, cone_angle_deg=9)
    phantom_recon_4 = create_phantom_and_reconstruction(
        seed=4, rot_deg_height=0, rot_deg_width=8, depth_sample_interval=20, cone_angle_deg=9)
    phantom_recon_5 = create_phantom_and_reconstruction(
        seed=4, rot_deg_height=0, rot_deg_width=10, depth_sample_interval=20, cone_angle_deg=9)
    phantom_recon_6 = create_phantom_and_reconstruction(
        seed=4, rot_deg_height=0, rot_deg_width=12, depth_sample_interval=20, cone_angle_deg=9)

    # Vary ring tilt (width) with cone angle 15°
    phantom_recon_7 = create_phantom_and_reconstruction(
        seed=4, rot_deg_height=0, rot_deg_width=6, depth_sample_interval=20, cone_angle_deg=15)
    phantom_recon_8 = create_phantom_and_reconstruction(
        seed=4, rot_deg_height=0, rot_deg_width=8, depth_sample_interval=20, cone_angle_deg=15)
    phantom_recon_9 = create_phantom_and_reconstruction(
        seed=4, rot_deg_height=0, rot_deg_width=10, depth_sample_interval=20, cone_angle_deg=15)
    phantom_recon_10 = create_phantom_and_reconstruction(
        seed=4, rot_deg_height=0, rot_deg_width=12, depth_sample_interval=20, cone_angle_deg=15)

    # Vary ring tilt (height) with cone angle 9°
    phantom_recon_11 = create_phantom_and_reconstruction(
        seed=4, rot_deg_height=6, rot_deg_width=0, depth_sample_interval=20, cone_angle_deg=9)
    phantom_recon_12 = create_phantom_and_reconstruction(
        seed=4, rot_deg_height=8, rot_deg_width=0, depth_sample_interval=20, cone_angle_deg=9)
    phantom_recon_13 = create_phantom_and_reconstruction(
        seed=4, rot_deg_height=10, rot_deg_width=0, depth_sample_interval=20, cone_angle_deg=9)
    phantom_recon_14 = create_phantom_and_reconstruction(
        seed=4, rot_deg_height=12, rot_deg_width=0, depth_sample_interval=20, cone_angle_deg=9)

    # Vary ring tilt (height) with cone angle 15°
    phantom_recon_15 = create_phantom_and_reconstruction(
        seed=4, rot_deg_height=6, rot_deg_width=0, depth_sample_interval=20, cone_angle_deg=15)
    phantom_recon_16 = create_phantom_and_reconstruction(
        seed=4, rot_deg_height=8, rot_deg_width=0, depth_sample_interval=20, cone_angle_deg=15)
    phantom_recon_17 = create_phantom_and_reconstruction(
        seed=4, rot_deg_height=10, rot_deg_width=0, depth_sample_interval=20, cone_angle_deg=15)
    phantom_recon_18 = create_phantom_and_reconstruction(
        seed=4, rot_deg_height=12, rot_deg_width=0, depth_sample_interval=20, cone_angle_deg=15)

    # Select which experiment sets to run
    exp1 = True  # Ring width tilt at cone angle 9°
    exp2 = True   # Ring width tilt at cone angle 15°
    exp3 = True  # Ring height tilt at cone angle 9°
    exp4 = True  # Ring height tilt at cone angle 15°

    if exp1:
        # Experiment 1: Ring width tilt, cone 9°
        set1 = (phantom_recon_1, phantom_recon_3, phantom_recon_4, phantom_recon_5, phantom_recon_6)
        set1_labels = [f"Tree-ring Width Tilt {x}°" for x in (0, 6, 8, 10, 12)]
        results1 = [find_best_reconstruction_depth(x) for x in set1]
        results1_r, results1_r_limited, results1_ssim = zip(*results1)

        method_labels = ["Weighted Radial Correlation Over All Angles",
                         "Weighted Radial Correlation Over Longest Angles",
                         "SSIM"]

        plot_reconstruction_quality_summary(results1[0], method_labels,
                                            title="Reconstruction Quality Metrics Comparison\n(Cone Angle 9°, No Tree-ring Tilt)",
                                            ylim=(0.6, 1))
        plot_reconstruction_quality_summary(results1_r, set1_labels,
                                            title="Weighted Radial Correlation Over All Angles (Cone Angle 9°)")
        plot_reconstruction_quality_summary(results1_r_limited, set1_labels,
                                            title="Weighted Radial Correlation Over Longest Angles (Cone Angle 9°)")

    if exp2:
        # Experiment 2: Ring width tilt, cone 15°
        set2 = (phantom_recon_2, phantom_recon_7, phantom_recon_8, phantom_recon_9, phantom_recon_10)
        set2_labels = [f"Tree-ring Width Tilt {x}°" for x in (0, 6, 8, 10, 12)]
        results2 = [find_best_reconstruction_depth(x) for x in set2]
        results2_r, results2_r_limited, results2_ssim = zip(*results2)

        method_labels = ["Weighted Radial Correlation Over All Angles",
                         "Weighted Radial Correlation Over Longest Angles",
                         "SSIM"]

        plot_reconstruction_quality_summary(results2[0], method_labels,
                                            title="Reconstruction Quality Metrics Comparison\n(Cone Angle 15°, No Tree-ring Tilt)",
                                            ylim=(0.6, 1))
        plot_reconstruction_quality_summary(results2_r, set2_labels,
                                            title="Weighted Radial Correlation Over All Angles (Cone Angle 15°)")
        plot_reconstruction_quality_summary(results2_r_limited, set2_labels,
                                            title="Weighted Radial Correlation Over Longest Angles (Cone Angle 15°)")

    if exp3:
        # Experiment 3: Ring height tilt, cone 9°
        set3 = (phantom_recon_1, phantom_recon_11, phantom_recon_12, phantom_recon_13, phantom_recon_14)
        set3_labels = [f"Tree-ring Height Tilt {x}°" for x in (0, 6, 8, 10, 12)]
        results3 = [find_best_reconstruction_depth(x) for x in set3]
        results3_r, results3_r_limited, results3_ssim = zip(*results3)

        plot_reconstruction_quality_summary(results3_r, set3_labels,
                                            title="Weighted Radial Correlation Over All Angles (Cone Angle 9°)",
                                            suffix="height")
        plot_reconstruction_quality_summary(results3_r_limited, set3_labels,
                                            title="Weighted Radial Correlation Over Longest Angles (Cone Angle 9°)",
                                            suffix="height")

    if exp4:
        # Experiment 4: Ring height tilt, cone 15°
        set4 = (phantom_recon_2, phantom_recon_15, phantom_recon_16, phantom_recon_17, phantom_recon_18)
        set4_labels = [f"Tree-ring Height Tilt {x}°" for x in (0, 6, 8, 10, 12)]
        results4 = [find_best_reconstruction_depth(x) for x in set4]
        results4_r, results4_r_limited, results4_ssim = zip(*results4)

        plot_reconstruction_quality_summary(results4_r, set4_labels,
                                            title="Weighted Radial Correlation Over All Angles (Cone Angle 15°)",
                                            suffix="height")
        plot_reconstruction_quality_summary(results4_r_limited, set4_labels,
                                            title="Weighted Radial Correlation Over Longest Angles (Cone Angle 15°)",
                                            suffix="height")
