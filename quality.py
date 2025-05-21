import numpy as np
from scipy.ndimage import map_coordinates
from phantom import generate_wood_block, interactive_slice_viewer
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr


def compute_max_radius_for_angle(image, center, theta):
    height, width = image.shape
    x0, y0 = center
    dx, dy = np.cos(theta), np.sin(theta)
    max_steps = np.inf
    if dx != 0:
        max_steps = min(max_steps, (width - 1 - x0) / dx if dx > 0 else -x0 / dx)
    if dy != 0:
        max_steps = min(max_steps, (height - 1 - y0) / dy if dy > 0 else -y0 / dy)
    return max_steps


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


def extract_radial_profile(image, center, theta, pixel_step=0.5):
    x0, y0 = center
    max_radius = compute_max_radius_for_angle(image, center, theta)
    radii = np.arange(0, max_radius, pixel_step)
    x_coords = x0 + radii * np.cos(theta)
    y_coords = y0 + radii * np.sin(theta)
    profile = map_coordinates(image, [y_coords, x_coords], order=1)
    return profile

# --- Utility Functions ---


def moving_average(data, window_size=5):
    """Simple moving average for detrending."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')


def baillie_pilcher_standardize(series):
    """Standardize by dividing by a smoothed version (Baillie-Pilcher style)."""
    smooth = moving_average(series, window_size=5)
    return series / smooth


def bandpass_filter(data, lowcut=0.05, highcut=0.3, fs=1.0, order=2):
    """Bandpass Butterworth filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# --- Metric Computations ---


def compute_tbp(series1, series2):
    """Compute TBP statistic and correlation r."""
    min_len = min(len(series1), len(series2))
    series1 = series1[:min_len]
    series2 = series2[:min_len]

    s1 = baillie_pilcher_standardize(np.array(series1))
    s2 = baillie_pilcher_standardize(np.array(series2))

    f1 = bandpass_filter(s1)
    f2 = bandpass_filter(s2)

    r, _ = pearsonr(f1, f2)
    n = len(f1)
    tbp = 100.0 if abs(r) == 1.0 else min(100, (r * np.sqrt(n - 2)) / np.sqrt(1 - r**2))
    return tbp, r


# Example usage:
if __name__ == "__main__":
    image3d, slices_c = generate_wood_block(seed=4, rot_deg_height=5, rot_deg_width=10)
    interactive_slice_viewer(image3d)

    baseline_theta = compute_angle_for_max_radius(image3d[0], slices_c[0])
    prof_baseline = extract_radial_profile(image3d[0], slices_c[0], theta=baseline_theta)
    r_all = []
    prof_lengths = []
    for deg in range(360):
        theta = np.deg2rad(deg)
        prof = extract_radial_profile(image3d[0], slices_c[0], theta=theta)
        prof_lengths.append(len(prof))
        tbp, r = compute_tbp(prof_baseline, prof)
        r_all.append(r)

    # Normalize profile lengths to match the scale of r values
    max_len = max(prof_lengths)
    norm_lengths = [2 * (l / max_len) - 1 for l in prof_lengths]  # Scaled to [-1, 1]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.set_rlim(-1, 1)

    angles = np.deg2rad(np.arange(360))
    r_all_np = np.array(r_all)
    norm_lengths_np = np.array(norm_lengths)

    ax.plot(angles, r_all_np, color='blue', label='Correlation (r)')
    ax.plot(angles, norm_lengths_np, color='orange', label='Profile length (normalized)')
    ax.fill(angles, r_all_np, color='skyblue', alpha=0.3)

    ax.set_theta_zero_location("E")  # 0° at the right
    ax.set_theta_direction(-1)        # Counterclockwise
    ax.set_title("Radial Correlation vs. Profile Length")
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

    plt.show()
