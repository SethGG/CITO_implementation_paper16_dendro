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


def extract_radial_profile(image, center, theta, radius, pixel_step):
    x0, y0 = center
    radii = np.arange(0, radius, pixel_step)
    x_coords = x0 + radii * np.cos(theta)
    y_coords = y0 + radii * np.sin(theta)
    profile = map_coordinates(image, [y_coords, x_coords], order=1)
    return profile

# Step 2: Derivative of radial profile to find ring boundaries


def find_ring_boundaries(profile, smoothing_window=10, width=5, prominence=0.8):
    derivative = savgol_filter(profile, window_length=smoothing_window, polyorder=1, deriv=1)
    peaks_max, _ = find_peaks(derivative, width=width, prominence=prominence * np.mean(np.abs(derivative)))
    peaks_min, _ = find_peaks(-derivative, width=width, prominence=prominence * np.mean(np.abs(derivative)))
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


# Step 4: Compute TBP

def compute_tbp_bp73(series1, series2):
    min_len = min(len(series1), len(series2))
    s1 = bp73_standardize(series1[:min_len])
    s2 = bp73_standardize(series2[:min_len])

    if len(s1) < 2 or len(s2) < 2:
        return np.nan, np.nan  # not enough data for correlation

    r, _ = pearsonr(s1, s2)
    n = len(s1)
    tbp = 100.0 if abs(r) == 1.0 else min(100, (r * np.sqrt(n - 2)) / np.sqrt(1 - r**2))
    return tbp, r

# Helper


def plot_peak_detection(bl_derivative, bl_peaks, derivative, peaks):
    fig = plt.figure(figsize=(12, 6))
    ax0 = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)

    ax0.plot(bl_derivative)
    ax0.plot(bl_peaks, bl_derivative[bl_peaks], 'x')
    ax0.set_title("Phantom profile")

    ax1.plot(derivative)
    ax1.plot(peaks, derivative[peaks], 'x')
    ax1.set_title("Reconstuction profile")

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
    bl_profile = extract_radial_profile(image, center, bl_theta, bl_radius, pixel_step=0.5)
    # extract the ring edge locations anc calculate ring widths
    bl_derivative, bl_peaks, bl_ring_widths = find_ring_boundaries(bl_profile)
    # standardize the ring width series
    bl_standard_ring_widths = bp73_standardize(bl_ring_widths)

    # Test on a noisy image as reconstruction placeholder
    image_recon = gaussian_filter(image, sigma=5)

    r_all = []
    num_rings = []
    for deg in range(360):
        theta = np.deg2rad(deg)
        radius = compute_radius_for_angle(image_recon, center, theta)
        profile = extract_radial_profile(image_recon, center, theta, radius, pixel_step=0.5)
        derivative, peaks, ring_widths = find_ring_boundaries(profile)
        standard_ring_widths = bp73_standardize(ring_widths)
        num_rings.append(len(ring_widths))
        tbp, r = compute_tbp_bp73(bl_standard_ring_widths, standard_ring_widths)
        if r < 0.6:
            plot_peak_detection(bl_derivative, bl_peaks, derivative, peaks)
        r_all.append(r)

    fig = plt.figure(figsize=(12, 10))
    ax0 = fig.add_subplot(221)
    ax3 = fig.add_subplot(223)
    ax1 = fig.add_subplot(222, polar=True)
    ax2 = fig.add_subplot(224, polar=True)

    angles = np.deg2rad(np.arange(360))
    r_all_np = np.array(r_all)

    ax1.plot(angles, r_all_np, color='blue', label='Correlation (r)')
    ax1.fill(angles, r_all_np, color='skyblue', alpha=0.3)
    ax1.set_theta_zero_location("E")  # 0° at the right
    ax1.set_theta_direction(-1)        # Counterclockwise
    ax1.set_title("Radial correlation of ring widths to baseline profile")
    ax1.legend(loc='upper right')

    ax2.plot(angles, num_rings, color='orange', label='Number of rings')
    ax2.fill(angles, num_rings, color='moccasin', alpha=0.3)
    ax2.set_theta_zero_location("E")  # 0° at the right
    ax2.set_theta_direction(-1)        # Counterclockwise
    ax2.set_title("Radial number of tree-rings")
    ax2.legend(loc='upper right')

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
