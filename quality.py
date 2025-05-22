import numpy as np
from scipy.ndimage import map_coordinates
from phantom import generate_wood_block, interactive_slice_viewer
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter

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


def find_ring_boundaries(profile, smoothing_window=20, prominence=0.01):
    derivative = savgol_filter(profile, window_length=smoothing_window, polyorder=1, deriv=1)
    peaks_max, _ = find_peaks(derivative, prominence=prominence*np.mean(np.abs(derivative)))
    peaks_min, _ = find_peaks(-derivative, prominence=prominence*np.mean(np.abs(derivative)))
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
            standardized.append(val)
    return np.array(standardized)


# Step 4: Compute TBP anf GL

def compute_tbp_bp73(series1, series2):
    min_len = min(len(series1), len(series2))
    s1 = series1[:min_len]
    s2 = series2[:min_len]

    if len(s1) < 2 or len(s2) < 2:
        return np.nan, np.nan  # not enough data for correlation

    r, _ = pearsonr(s1, s2)
    n = len(s1)
    tbp = 100.0 if abs(r) == 1.0 else min(100, (r * np.sqrt(n - 2)) / np.sqrt(1 - r**2))
    return tbp, r


def compute_gl(series1, series2):
    min_len = min(len(series1), len(series2))
    s1 = np.diff(series1[:min_len])
    s2 = np.diff(series2[:min_len])
    valid = (s1 != 0) & (s2 != 0)
    agree = (np.sign(s1[valid]) == np.sign(s2[valid])).sum()
    return 100 * agree / valid.sum() if valid.sum() > 0 else np.nan

# Helper functions to compare ring detection


def plot_peak_detection(deg_1, profile_1, derivative_1, peaks_1, ring_widths_1,
                        deg_2, profile_2, derivative_2, peaks_2, ring_widths_2):
    fig = plt.figure(figsize=(12, 6))
    ax0 = fig.add_subplot(321)
    ax1 = fig.add_subplot(322)
    ax2 = fig.add_subplot(323)
    ax3 = fig.add_subplot(324)
    ax4 = fig.add_subplot(325)
    ax5 = fig.add_subplot(326)

    ax0.plot(profile_1)
    ax1.plot(profile_2)

    ax2.plot(derivative_1)
    ax2.plot(peaks_1, derivative_1[peaks_1], 'x')
    ax3.plot(derivative_2)
    ax3.plot(peaks_2, derivative_2[peaks_2], 'x')

    ax4.plot(ring_widths_1)
    ax5.plot(ring_widths_2)

    plt.show()

# Plot radial correlation and ring numbers in recontruction


def plot_radial_correlation(angles, bl_theta, bl_radius, all_r, all_gl, all_num_rings, image, center, image_recon):
    angles = angles - bl_theta

    fig = plt.figure(figsize=(12, 10))
    ax0 = fig.add_subplot(221)
    ax3 = fig.add_subplot(222)
    ax1 = fig.add_subplot(234, polar=True)
    ax4 = fig.add_subplot(235, polar=True)
    ax2 = fig.add_subplot(236, polar=True)

    ax1.plot(angles, all_r, color='blue', label='Correlation (r)')
    ax1.fill(angles, all_r, color='skyblue', alpha=0.3)
    ax1.set_theta_zero_location("N", offset=np.rad2deg(-bl_theta)+270)  # 0° at the right
    ax1.set_theta_direction(-1)        # Counterclockwise
    # ax1.set_title("Radial correlation of ring widths to baseline profile")
    ax1.legend(loc='upper right')

    ax2.plot(angles, all_num_rings, color='orange', label='Number of rings')
    ax2.fill(angles, all_num_rings, color='moccasin', alpha=0.3)
    ax2.set_theta_zero_location("N", offset=np.rad2deg(-bl_theta)+270)  # 0° at the right
    ax2.set_theta_direction(-1)        # Counterclockwise
    # ax2.set_title("Radial number of tree-rings")
    ax2.legend(loc='upper right')

    ax4.plot(angles, all_gl, color='green', label='Gleichläufigkeit (GL)')
    ax4.fill(angles, all_gl, color='limegreen', alpha=0.3)
    ax4.set_theta_zero_location("N", offset=np.rad2deg(-bl_theta)+270)  # 0° at the right
    ax4.set_theta_direction(-1)        # Counterclockwise
    # ax1.set_title("Radial correlation of ring widths to baseline profile")
    ax4.legend(loc='upper right')

    ax0.imshow(image, cmap='gray')
    ax0.set_title("Phantom ground truth")
    # Compute end point of the max radius line
    x0, y0 = center
    x1 = x0 + bl_radius * np.cos(bl_theta)
    y1 = y0 + bl_radius * np.sin(bl_theta)
    # Plot the line from center to (x1, y1)
    ax0.plot([x0, x1], [y0, y1], color='red', linestyle='-', linewidth=2, label='Max radius line (Baseline profile)')
    ax0.legend(loc='lower right')

    ax3.imshow(image_recon, cmap='gray')
    ax3.set_title("Reconstruction")

    plt.show()

# Dynamic programming for best segmented correlation


def dp_segmented_correlation_precomputed(reference, recon_segments, min_segment_len=20, max_segment_len=40):
    N = len(reference)
    A = len(recon_segments)
    dp = [-np.inf] * (N + 1)
    backtrack = [None] * (N + 1)
    dp[0] = 0.0

    # Precompute correlations for all segments
    print("Precompute correlations for all segments")
    corr = {}
    for j in range(N):
        for i in range(j + min_segment_len, min(N, j + max_segment_len) + 1):
            ref_seg = reference[j:i]
            if len(ref_seg) < 2:
                continue
            for angle_idx, recon_seg in enumerate(recon_segments):
                if len(recon_seg) < i:  # Must be able to slice j:i
                    continue
                recon_sub = recon_seg[j:i]
                if len(recon_sub) < 2:
                    continue
                r, _ = pearsonr(ref_seg, recon_sub)
                corr[(j, i, angle_idx)] = r

    # Run DP using the precomputed correlations
    print("Run DP using the precomputed correlations")
    for i in range(min_segment_len, N + 1):
        for j in range(max(0, i - max_segment_len), i - min_segment_len + 1):
            for angle_idx in range(A):
                r = corr.get((j, i, angle_idx), None)
                if r is None:
                    continue
                weight = (i - j) / N
                score = dp[j] + weight * r
                if score > dp[i]:
                    dp[i] = score
                    backtrack[i] = (j, i, angle_idx, r)

    # Reconstruct best path
    print("Reconstruct best path")
    i = N
    best_segments = []
    while i > 0 and backtrack[i] is not None:
        seg = backtrack[i]
        best_segments.insert(0, seg)
        i = seg[0]

    return dp[N], best_segments

# Plot best tree ring segments found by DP on the image


def plot_segment_marks_on_image(image, center, segments, angles, ring_boundaries_all, pixel_step=0.5):
    """
    Marks ring boundaries from selected segments on the reconstruction image.

    Parameters:
        image: 2D numpy array (reconstruction image)
        center: (x0, y0) tuple of ring center
        segments: list of tuples (start, end, angle_idx, r) from DP
        angles: list/array of angles in radians, where index matches angle_idx
        ring_boundaries_all: list of peaks arrays from each angle's profile (i.e., all_peaks)
        pixel_step: spacing between radial samples (used in extract_radial_profile)
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap='gray')
    ax.set_title("Rings used in optimal correlation segments")

    x0, y0 = center

    for (start, end, angle_idx, _) in segments:
        theta = angles[angle_idx]
        ring_peaks = ring_boundaries_all[angle_idx]

        # Account for +2 shift due to bp73_standardize
        ring_start = start + 2
        ring_end = end + 2

        # Main used peaks
        main_peaks = ring_peaks[ring_start:ring_end]
        for r in main_peaks:
            radius = r * pixel_step
            x = x0 + radius * np.cos(theta)
            y = y0 + radius * np.sin(theta)
            ax.plot(x, y, 'x', color='lime', markersize=6)

        # Padding peaks (before and after)
        pad_before = ring_peaks[max(0, ring_start - 2):ring_start]
        pad_after = ring_peaks[ring_end:ring_end + 2]

        for r in np.concatenate([pad_before, pad_after]):
            radius = r * pixel_step
            x = x0 + radius * np.cos(theta)
            y = y0 + radius * np.sin(theta)
            ax.plot(x, y, 'x', color='magenta', markersize=6)

    plt.show()


# Example usage:
if __name__ == "__main__":
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

    all_num_rings = [len(x) for x in all_ring_widths]

    plot_radial_correlation(angles, bl_theta, bl_radius, all_r, all_gl, all_num_rings, image, center, image_recon)

    # score, segments = dp_segmented_correlation_precomputed(
    #     reference=bl_standard_ring_widths,
    #     recon_segments=all_standard_ring_widths
    # )

    # print(f"DP Optimal Weighted Correlation Score: {score:.3f}")
    # for start, end, angle_idx, r in segments:
    #     print(f"Rings {start}-{end} → angle {angle_idx} → r = {r:.3f}")

    # plot_segment_marks_on_image(
    #     image=image_recon,
    #     center=center,
    #     segments=segments,
    #     angles=angles,
    #     ring_boundaries_all=all_peaks,
    #     pixel_step=0.5
    # )
