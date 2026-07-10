import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.ndimage import convolve1d, gaussian_filter1d
from sklearn.manifold import Isomap

def plot_head_direction_tuning(
    t,
    angle_data,
    spike_array_cell_i,
    ax=None,
    figsize=(2, 2),
    dpi=100,
):
    """
    Plot the head direction tuning curve for one neuron on polar axes.

    By default, this function creates its own figure so it can be used in the
    same style as the raster helper. If an existing polar axis is provided, the
    plot is drawn there instead.

    Parameters:
    t : np.array
        Time array.
    angle_data : np.array
        Array of session head direction angles in radians.
    spike_array_cell_i : np.array
        Spike array for one cell, aligned with t and angle_data.
    ax : matplotlib polar axis or None
        Existing polar axis to draw on. If None, create a new figure and axis.
    figsize : tuple
        Figure size used only when `ax` is None.
    dpi : int
        Figure resolution used only when `ax` is None.

    Returns:
    fig, ax
        Matplotlib figure and polar axis.
    """

    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, polar=True)
    else:
        fig = ax.figure

    hd_bins = 60
    boxcar_size = 3
    pos_tb = np.median(np.diff(t))

    # Calculate the head direction angle when spike_array_cell_i is not zero
    angle4spike = angle_data[spike_array_cell_i != 0]

    # Calculate histograms for session and cell head directions
    hd1, _ = np.histogram(angle_data, bins=hd_bins, range=(-np.pi, np.pi))
    hd2, _ = np.histogram(angle4spike, bins=hd_bins, range=(-np.pi, np.pi))

    # Boxcar filter
    boxcar_filter = np.ones(boxcar_size) / boxcar_size

    # Apply boxcar filter
    hd1_filtered = convolve1d(hd1, boxcar_filter, mode='wrap')
    hd2_filtered = convolve1d(hd2, boxcar_filter, mode='wrap')

    # Convert session HD to time, i.e. dwelling time in each HD bin
    hd1_time = hd1_filtered * pos_tb

    # Calculate HD firing rate
    hd3 = hd2_filtered / hd1_time

    # Normalize session HD
    hd1_normalized = hd1_time / np.max(hd1_time)

    # Normalize cell HD firing rate
    hd3_normalized = hd3 / np.nanmax(hd3)
    hd3_normalized = hd3_normalized.flatten()

    # Close the loop by appending the first element to the end
    theta = np.linspace(-np.pi, np.pi, hd_bins, endpoint=False)
    theta = np.append(theta, theta[0])
    hd1_normalized = np.append(hd1_normalized, hd1_normalized[0])
    hd3_normalized = np.append(hd3_normalized, hd3_normalized[0])

    # Plot the session head direction with shading
    ax.plot(theta, hd1_normalized, label='Session Head Direction', color='gray')
    ax.fill_between(theta, 0, hd1_normalized, facecolor='gray', alpha=0.2)

    # Plot the cell head direction firing rate
    ax.plot(theta, hd3_normalized, label='Cell Head Direction Firing Rate', color='#38c7ff')
    ax.fill_between(theta, 0, hd3_normalized, facecolor='#38c7ff', alpha=0.5)

    # Keep 0 90 180 270 as the xticks
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax.set_xticklabels(['0', '90', '180', '270'])

    # Remove yticks
    ax.set_yticks([])

    # Calculate the preferred direction which corresponds to the peak of the tuning curve
    hd_peakfr = theta[np.argmax(hd3_normalized)]

    # Add a line to indicate the preferred direction
    ax.plot([hd_peakfr, hd_peakfr], [0, 1], color='red', linestyle='--', linewidth=1)

    return fig, ax


def plot_hd_raster_with_trace(
    t,
    angle_data,
    spike_array,
    t_start=None,
    window_duration=100,
    cell_ids=None,
    figsize=(8, 5),
    raster_linelength=0.2,
    raster_linewidth=0.5,
    trace_color='#38c7ff',
):
    """
    Plot head direction and spike raster in a 2-by-1 layout with aligned x-axis.

    The top panel shows head direction over time.
    The bottom panel shows the raster plot for a set of neurons.

    Parameters:
    t : np.ndarray
        Time vector.
    angle_data : np.ndarray
        Head direction angles in radians, aligned with `t`.
    spike_array : np.ndarray
        Spike matrix with shape (n_neurons, n_timepoints).
        Nonzero values are treated as spikes.
    t_start : float or None
        Start time of the plotting window. If None, use the first time point.
    window_duration : float
        Duration of the plotting window in seconds.
    cell_ids : array-like or None
        Indices of neurons to include in the raster plot.
        If None, plot all neurons.
    figsize : tuple
        Figure size passed to matplotlib.
    raster_linelength : float
        Vertical length of each raster tick mark.
    raster_linewidth : float
        Line width of each raster tick mark.
    trace_color : str
        Color of the head direction trace.

    Returns:
    fig, (ax1, ax2)
        Matplotlib figure and axes.
    """

    if t_start is None:
        t_start = t[0]
    t_end = t_start + window_duration

    if cell_ids is None:
        cell_ids = np.arange(spike_array.shape[0])
    else:
        cell_ids = np.asarray(cell_ids)

    # Select the requested time window for the head direction trace.
    time_mask = (t >= t_start) & (t <= t_end)
    t_window = t[time_mask]
    angle_window = angle_data[time_mask].copy()

    # Break the line at wrap-around points so -pi and pi are not connected.
    angle_diff = np.abs(np.diff(angle_window))
    jump_idx = np.where(angle_diff > np.pi)[0] + 1
    angle_window[jump_idx] = np.nan

    # Convert each neuron's spike train into a list of spike times in the window.
    spike_times_list = []
    for neuron_idx in cell_ids:
        spike_times = t[spike_array[neuron_idx] > 0]
        spike_times = spike_times[(spike_times >= t_start) & (spike_times <= t_end)]
        spike_times_list.append(spike_times)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={'height_ratios': [1, 3]},
    )

    # Top panel: head direction trace.
    ax1.plot(t_window, angle_window, color=trace_color, linewidth=1.5)
    ax1.set_ylabel('HD (rad)')
    ax1.set_title(f'Head Direction and Raster Plot ({int(t_start)} to {int(t_end)} s)')
    ax1.set_ylim(-np.pi, np.pi)
    ax1.set_yticks([-np.pi, 0, np.pi])
    ax1.set_yticklabels([r'$-\pi$', '0', r'$\pi$'])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Bottom panel: spike raster.
    ax2.eventplot(
        spike_times_list,
        colors='black',
        lineoffsets=np.arange(1, len(cell_ids) + 1),
        linelengths=raster_linelength,
        linewidths=raster_linewidth,
    )
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Neuron index')
    ax2.set_yticks(np.arange(1, len(cell_ids) + 1))
    ax2.set_yticklabels(np.arange(1, len(cell_ids) + 1))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_hd_isomap(
    t_binned,
    angle_binned,
    spike_binned,
    gap_threshold_factor=5,
    smooth_sd=0.4,
    low_percentile=1,
    high_percentile=99,
    max_points=10000,
    n_neighbors=12,
    n_components=3,
    figsize=(6, 5),
    point_size=5,
    alpha=0.85,
    cmap='hsv',
):
    """
    Plot a 2D Isomap embedding of population activity colored by head direction.

    This helper follows the same analysis steps as the notebook version:
    1. Find continuous recording segments.
    2. Use the longest segment.
    3. Smooth population activity and apply a square-root transform.
    4. Remove bins with extremely low or high population activity.
    5. Run Isomap and color the embedding by real head direction.

    Parameters:
    t_binned : np.ndarray
        Time vector for binned data.
    angle_binned : np.ndarray
        Head direction angle for each time bin, in radians.
    spike_binned : np.ndarray
        Binned spike counts with shape (n_cells, n_bins).
    gap_threshold_factor : float
        A gap larger than this factor times the median bin size starts a new segment.
    smooth_sd : float
        Gaussian smoothing standard deviation in seconds.
    low_percentile : float
        Lower population-activity percentile for filtering bins.
    high_percentile : float
        Upper population-activity percentile for filtering bins.
    max_points : int
        Maximum number of points to plot after optional subsampling.
    n_neighbors : int
        Number of neighbors for Isomap.
    n_components : int
        Number of Isomap dimensions to compute.
    figsize : tuple
        Figure size for the embedding plot.
    point_size : float
        Scatter marker size.
    alpha : float
        Scatter marker transparency.
    cmap : str
        Matplotlib colormap for head direction coloring.

    Returns:
    results : dict
        Dictionary containing the figure, axes, embedding, selected segment,
        filtered angles, filtered times, and intermediate arrays.
    """

    # Split the recording into continuous segments if large gaps are present.
    bin_size = np.median(np.diff(t_binned))
    dt_binned = np.diff(t_binned)
    gap_idx = np.where(dt_binned > gap_threshold_factor * bin_size)[0]

    segment_starts = np.r_[0, gap_idx + 1]
    segment_ends = np.r_[gap_idx + 1, len(t_binned)]
    segment_lengths = segment_ends - segment_starts
    seg_id = np.argmax(segment_lengths)

    start = segment_starts[seg_id]
    end = segment_ends[seg_id]

    # Keep the longest continuous segment for manifold analysis.
    t_seg = t_binned[start:end]
    angle_seg = angle_binned[start:end]
    spike_seg = spike_binned[:, start:end]

    # Build population vectors with shape (n_bins, n_cells).
    X = spike_seg.T.astype(float)

    # Smooth in time and apply a square-root transform to compress large values.
    seg_bin_size = np.median(np.diff(t_seg))
    sigma_bins = smooth_sd / seg_bin_size
    X_smooth = gaussian_filter1d(X, sigma=sigma_bins, axis=0, mode='reflect')
    X_sqrt = np.sqrt(X_smooth)

    # Remove bins with extreme total population activity.
    pop_activity = X_sqrt.sum(axis=1)
    low_q = np.percentile(pop_activity, low_percentile)
    high_q = np.percentile(pop_activity, high_percentile)
    good_bins = (
        (pop_activity > low_q) &
        (pop_activity < high_q) &
        np.isfinite(pop_activity)
    )

    X_iso_input = X_sqrt[good_bins]
    angle_iso = angle_seg[good_bins]
    t_iso = t_seg[good_bins]

    # Optionally subsample to keep plotting and Isomap manageable.
    if X_iso_input.shape[0] > max_points:
        plot_idx = np.linspace(0, X_iso_input.shape[0] - 1, max_points).astype(int)
    else:
        plot_idx = np.arange(X_iso_input.shape[0])

    X_plot = X_iso_input[plot_idx]
    angle_plot = angle_iso[plot_idx]
    t_plot = t_iso[plot_idx]

    # Compute the manifold embedding.
    isomap = Isomap(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric='euclidean',
    )
    X_isomap = isomap.fit_transform(X_plot)
    X_isomap = X_isomap - X_isomap.mean(axis=0)

    # Color points by real head direction.
    angle_color = (angle_plot + np.pi) / (2 * np.pi)

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(
        X_isomap[:, 0],
        X_isomap[:, 1],
        c=angle_color,
        cmap=cmap,
        s=point_size,
        alpha=alpha,
        linewidths=0,
    )

    ax.set_aspect('equal', adjustable='box')
    ax.set_title('HD population Isomap')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Keep endpoint labels slightly inside the colorbar so they do not get clipped.
    cbar = plt.colorbar(sc, ax=ax, ticks=[0.02, 0.5, 0.98])
    cbar.ax.set_yticklabels(['-180', '0', '180'])
    cbar.set_label('Real head direction')

    plt.tight_layout()

    return {
        'fig': fig,
        'ax': ax,
        'embedding': X_isomap,
        'angle_plot': angle_plot,
        'time_plot': t_plot,
        'segment_id': seg_id,
        'segment_start': start,
        'segment_end': end,
        'segment_starts': segment_starts,
        'segment_ends': segment_ends,
        'good_bins': good_bins,
        'plot_idx': plot_idx,
        'X_plot': X_plot,
    }


def plot_ring_connectivity_layout(
    preferred_angles,
    weights,
    local_excitation=None,
    broad_inhibition=None,
    neuron_idx=64,
    figsize=(11, 5),
    neuron_color='grey',
    weight_cmap='coolwarm',
    neuron_size=4,
):
    """
    Plot the ring layout of neurons together with connectivity summaries.

    This helper keeps the notebook focused on model ideas rather than plotting
    details. The left panel shows neurons arranged by preferred direction on a
    ring. If `local_excitation` and `broad_inhibition` are provided, the middle
    panel shows one neuron's excitation and inhibition profiles, and the right
    panel shows the recurrent connectivity matrix.

    Parameters:
    preferred_angles : np.ndarray
        Preferred angle of each simulated neuron, in radians.
    weights : np.ndarray
        Recurrent weight matrix with shape (n_neurons, n_neurons).
    local_excitation : np.ndarray or None
        Excitatory component of the recurrent weights. If provided together
        with `broad_inhibition`, a single-neuron profile plot is added.
    broad_inhibition : np.ndarray or None
        Inhibitory component of the recurrent weights.
    neuron_idx : int
        Which neuron to use for the profile plot in the middle panel.
    figsize : tuple
        Figure size passed to matplotlib.
    neuron_color : str
        Color used for neurons on the ring.
    weight_cmap : str
        Colormap used for the recurrent weight matrix.
    neuron_size : float
        Marker size for neurons on the ring.

    Returns:
    fig, axes
        Matplotlib figure and axes.
    """

    show_profiles = (
        local_excitation is not None and
        broad_inhibition is not None
    )

    fig = plt.figure(figsize=figsize)

    # Left panel: neurons arranged around the ring.
    if show_profiles:
        ax1 = fig.add_subplot(1, 3, 1, polar=True)
    else:
        ax1 = fig.add_subplot(1, 2, 1, polar=True)

    ax1.set_theta_zero_location('E')
    ax1.set_theta_direction(-1)
    ax1.scatter(
        preferred_angles,
        np.ones_like(preferred_angles),
        s=neuron_size,
        color=neuron_color,
    )
    ax1.set_title(f'{len(preferred_angles)} Simulated Neurons Arranged on a Ring')
    ax1.set_rticks([])
    ax1.set_ylim(0,1.2)
    ax1.spines['polar'].set_visible(False)
    ax1.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax1.set_xticklabels(['0', '90', '180', '270'])

    if show_profiles:
        # Middle panel: one neuron's excitation and inhibition profile.
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(
            preferred_angles,
            local_excitation[neuron_idx],
            linewidth=2,
            color='tab:red',
            label='Local excitation',
        )
        ax2.plot(
            preferred_angles,
            broad_inhibition[neuron_idx],
            linewidth=2,
            color='tab:blue',
            label='Broad inhibition',
        )
        ax2.axvline(
            preferred_angles[neuron_idx],
            color='black',
            linestyle='--',
            linewidth=1,
        )
        ax2.set_xlabel('Preferred angle (rad)')
        ax2.set_ylabel('Connection strength')
        ax2.set_title(f'Connectivity Profile for Neuron {neuron_idx}')
        ax2.legend(frameon=False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # Right panel: recurrent weight matrix.
        ax3 = fig.add_subplot(1, 3, 3)
        im = ax3.imshow(weights, aspect='auto', cmap=weight_cmap)
        ax3.set_xlabel('Presynaptic neuron')
        ax3.set_ylabel('Postsynaptic neuron')
        ax3.set_title('Recurrent Weight Matrix')

        cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        cbar.set_label('Weight')
        axes = (ax1, ax2, ax3)
    else:
        # Right panel: recurrent weight matrix.
        ax2 = fig.add_subplot(1, 2, 2)
        im = ax2.imshow(weights, aspect='auto', cmap=weight_cmap)
        ax2.set_xlabel('Presynaptic neuron')
        ax2.set_ylabel('Postsynaptic neuron')
        ax2.set_title('Recurrent Weight Matrix')

        cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Weight')
        axes = (ax1, ax2)

    plt.tight_layout()
    return fig, axes


def plot_av_hd_schematic(
    W_cw,
    W_ccw,
    n_neurons=36,
    step=2,
    figsize=(10, 4),
    dpi=120,
    av_plus_color='#f5a623',
    hd_color='0.88',
    av_minus_color='#37cbe0',
    av_plus_label='AV+',
    hd_label='HD\nlayer',
    av_minus_label='AV-',
):
    """
    Plot a schematic using real AV-to-HD weight matrices.

    The three populations are drawn as 1D rows spanning 0 to 360 degrees:
    - AV+ row on top
    - HD layer in the middle
    - AV- row on the bottom

    For each displayed AV neuron, the function finds the strongest target
    neuron in the supplied weight matrix and draws one arrow to that target.
    This makes the schematic reflect the actual matrices rather than a
    hand-specified fixed shift.

    For clarity, wrap-around links are not drawn specially; instead the
    strongest full-resolution target is mapped to the nearest displayed neuron.

    Returns:
    fig, ax
        Matplotlib figure and axis.
    """

    full_n_neurons = W_cw.shape[0]
    if W_cw.shape != W_ccw.shape:
        raise ValueError("W_cw and W_ccw must have the same shape.")

    x = np.linspace(0, 1, n_neurons)
    y_av_plus = 2.0
    y_hd = 1.0
    y_av_minus = 0.0
    source_idx = np.linspace(0, full_n_neurons - 1, n_neurons, dtype=int)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Draw three rows of neurons.
    ax.scatter(
        x,
        np.full_like(x, y_av_plus),
        s=220,
        facecolor=av_plus_color,
        edgecolor='0.3',
        linewidth=1.2,
        zorder=3,
    )
    ax.scatter(
        x,
        np.full_like(x, y_hd),
        s=220,
        facecolor=hd_color,
        edgecolor='0.3',
        linewidth=1.2,
        zorder=3,
    )
    ax.scatter(
        x,
        np.full_like(x, y_av_minus),
        s=220,
        facecolor=av_minus_color,
        edgecolor='0.3',
        linewidth=1.2,
        zorder=3,
    )

    # Row baselines.
    ax.plot([x[0], x[-1]], [y_av_plus, y_av_plus], color='0.65', lw=6, alpha=0.4, zorder=1)
    ax.plot([x[0], x[-1]], [y_hd, y_hd], color='0.65', lw=6, alpha=0.4, zorder=1)
    ax.plot([x[0], x[-1]], [y_av_minus, y_av_minus], color='0.65', lw=6, alpha=0.4, zorder=1)

    # Labels.
    ax.text(-0.08, y_av_plus, av_plus_label, color=av_plus_color, fontsize=16,
            va='center', ha='right', fontweight='bold')
    ax.text(-0.08, y_hd, hd_label, color='black', fontsize=16,
            va='center', ha='right')
    ax.text(-0.08, y_av_minus, av_minus_label, color='#1aa6c1', fontsize=16,
            va='center', ha='right', fontweight='bold')

    ax.text(x[0] - 0.01, y_av_plus + 0.18, '0°', fontsize=13, ha='center')
    ax.text(x[-1] + 0.01, y_av_plus + 0.18, '360°', fontsize=13, ha='center')
    ax.text(x[0] - 0.01, y_hd + 0.18, '0°', fontsize=13, ha='center')
    ax.text(x[-1] + 0.01, y_hd + 0.18, '360°', fontsize=13, ha='center')
    ax.text(x[0] - 0.01, y_av_minus - 0.18, '0°', fontsize=13, ha='center')
    ax.text(x[-1] + 0.01, y_av_minus - 0.18, '360°', fontsize=13, ha='center')

    def draw_matrix_projections(W, y_from, y_to, color, line_step, alpha=0.45, lw=1.5):
        display_idx = np.arange(0, n_neurons, line_step)

        for display_i in display_idx:
            full_i = source_idx[display_i]
            full_j = int(np.argmax(W[full_i]))
            display_j = int(np.argmin(np.abs(source_idx - full_j)))

            start = (x[display_i], y_from - 0.05 if y_from > y_to else y_from + 0.05)
            end = (x[display_j], y_to + 0.05 if y_from > y_to else y_to - 0.05)

            arrow = FancyArrowPatch(
                start,
                end,
                arrowstyle='->',
                mutation_scale=12,
                lw=lw,
                color=color,
                alpha=alpha,
                zorder=2,
            )
            ax.add_patch(arrow)

    draw_matrix_projections(W_cw, y_av_plus, y_hd, av_plus_color, step)
    draw_matrix_projections(W_ccw, y_av_minus, y_hd, av_minus_color, step)

    # Emphasize one example connection from each AV population.
    mid = n_neurons // 2
    mid_full = source_idx[mid]
    cw_target_full = int(np.argmax(W_cw[mid_full]))
    ccw_target_full = int(np.argmax(W_ccw[mid_full]))
    cw_target_display = int(np.argmin(np.abs(source_idx - cw_target_full)))
    ccw_target_display = int(np.argmin(np.abs(source_idx - ccw_target_full)))

    ax.add_patch(FancyArrowPatch(
        (x[mid], y_av_plus - 0.08),
        (x[cw_target_display], y_hd + 0.08),
        arrowstyle='->',
        mutation_scale=20,
        lw=4,
        color='#e58e00',
        alpha=0.95,
        zorder=4,
    ))
    ax.add_patch(FancyArrowPatch(
        (x[mid], y_av_minus + 0.08),
        (x[ccw_target_display], y_hd - 0.08),
        arrowstyle='->',
        mutation_scale=20,
        lw=4,
        color='#159fbc',
        alpha=0.95,
        zorder=4,
    ))

    ax.set_xlim(-0.12, 1.05)
    ax.set_ylim(-0.35, 2.35)
    ax.axis('off')
    plt.tight_layout()
    return fig, ax
