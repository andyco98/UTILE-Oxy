##IMPORTS AND INPUTS##

import os
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.colors as mcolors
from tracking_pipeline_complete import*  

global analysis_save_path

########## NEW BUBBLES VS TIME ##########

def plot_new_bubbles_vs_time(csv_path, case_study):
    """
    Analyze and plot the temporal evolution of new bubbles for a given dataset.

    Args:
        csv_path (str): The path to the combined CSV file from the tracking algorithm.
        case_study (str): Case details associated with the dataset (e.g., "1500mV").
    """


    

    # Create a folder for analysis plots in the same directory as the CSV file
    parent_directory = Path(csv_path).parent
    analysis_save_path = parent_directory /f"Analysis Plots for {case_study}"
    analysis_save_path.mkdir(parents=True, exist_ok=True)

    # Function to detect and keep only new bubble events (labels without parentheses)
    def filter_new_bubbles(df):
        # Keep only rows where label column does not contain parentheses
        df_new_bubbles = df[~df['label'].str.contains(r'\(.*\)', regex=True)].copy()
        # Remove duplicates to keep only unique new bubble events
        df_new_bubbles = df_new_bubbles.drop_duplicates(subset=['label'])
        return df_new_bubbles

    # Function to count new bubble events based on the frame number in Object ID
    def count_new_bubbles_per_frame(df_new_bubbles):
        # Extract the frame number from Object ID using regex
        df_new_bubbles['frame_number'] = df_new_bubbles['label'].str.extract(r'_frame(\d+)', expand=False).astype(int)
        # Count how many new bubbles occur per frame
        new_bubble_counts = df_new_bubbles.groupby('frame_number').size().reset_index(name='new_bubble_count')
        # Create a cumulative sum of new bubble events
        new_bubble_counts['cumulative_new_bubble_count'] = new_bubble_counts['new_bubble_count'].cumsum()
        return new_bubble_counts

    # Check if CSV file exists
    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} does not exist.")
        return

    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Filter and count new bubble events
    df_new_bubbles = filter_new_bubbles(df)
    new_bubble_counts = count_new_bubbles_per_frame(df_new_bubbles)

    if new_bubble_counts.empty:
        print(f"No new bubble events found in {csv_path}.")
        return

    # Plot the cumulative new bubble event trend
    plt.figure(figsize=(15, 9))
    plt.plot(
        new_bubble_counts['frame_number'],
        new_bubble_counts['cumulative_new_bubble_count'],
        label=f"{case_study}"
    )

    # Customize the plot
    plt.title(f"Temporal Evolution of New Bubbles for {case_study}", fontsize=28)
    plt.xlabel("Frame Number", fontsize=18)
    plt.ylabel("Cumulative New Bubble Count", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=18)

    # Save the plot
    plot_save_path = analysis_save_path / f" New Bubbles vs. Time {case_study}.png"
    plt.savefig(plot_save_path, format='png')
    print(f"Plot saved to {plot_save_path}")

    # Show the plot
    plt.show()

########## MERGING EVENTS VS TIME ##########

def plot_merging_events_vs_time(csv_path, case_study):
    """
    Analyze and plot the temporal evolution of merge events for a given dataset.

    Args:
        csv_path (str): The path to the combined CSV file from the tracking algorithm.
        case_study (str): Case details associated with the dataset (e.g., "1500mV", serpentine etc.).
    """
    
    
    

    # Create a folder for analysis plots in the same directory as the CSV file
    parent_directory = Path(csv_path).parent
    analysis_save_path = parent_directory / f"Analysis Plots for {case_study}"
    analysis_save_path.mkdir(parents=True, exist_ok=True)

    # Function to detect and keep only merge events (labels with '+')
    def filter_merge_events(df):
        # Keep only rows where label column contains '+', indicating a merge
        df_merge = df[df['label'].str.contains(r'\+\d+', regex=True)].copy()
        # Remove duplicates to keep only unique merge events
        df_merge = df_merge.drop_duplicates(subset=['label'])
        return df_merge

    # Function to count merge events based on the frame number in Object ID
    def count_merges_per_frame(df_merge):
        # Extract the frame number from Object ID using regex
        df_merge['frame_number'] = df_merge['label'].str.extract(r'_frame(\d+)', expand=False).astype(int)
        # Count how many merges occur per frame
        merge_counts = df_merge.groupby('frame_number').size().reset_index(name='merge_count')
        # Create a cumulative sum of merge events
        merge_counts['cumulative_merge_count'] = merge_counts['merge_count'].cumsum()
        return merge_counts

    # Check if CSV file exists
    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} does not exist.")
        return

    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Filter and count merge events
    df_merge = filter_merge_events(df)
    merge_counts = count_merges_per_frame(df_merge)

    if merge_counts.empty:
        print(f"No merge events found in {csv_path}.")
        return

    # Plot the cumulative merge event trend
    plt.figure(figsize=(15, 9))
    plt.plot(
        merge_counts['frame_number'],
        merge_counts['cumulative_merge_count'],
        label=f"{case_study}"
    )

    # Customize the plot
    plt.title(f"Temporal Evolution of Merge Events for {case_study}", fontsize=28)
    plt.xlabel("Frame Number", fontsize=18)
    plt.ylabel("Cumulative Merge Count", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=18)

    # Save the plot
    plot_save_path = analysis_save_path / f"Merge Events vs. Time {case_study}.png"
    plt.savefig(plot_save_path, format='png')
    print(f"Plot saved to {plot_save_path}")

    # Show the plot
    plt.show()

########## NEW BUBBLES EMERGING POINTS HEATMAP ##########

def generate_heatmap(csv_path, case_study):
    """
    Generates a heatmap of first occurrences of objects based on their centroids.

    Args:
        csv_path (str): Path to the CSV file containing tracking data.
        case_study (str): Case details associated with the dataset (e.g., "1500mV").
    """

    
    parent_directory = Path(csv_path).parent
    analysis_save_path = parent_directory / f"Analysis Plots for {case_study}"
    analysis_save_path.mkdir(parents=True, exist_ok=True)

    # Optimized function to draw circles using vectorized operations
    def draw_circle(heatmap, x, y, radius=15, value=0.4):
        """Draws a circle on the heatmap centered at (x, y)."""
        xx, yy = np.ogrid[:heatmap.shape[0], :heatmap.shape[1]]
        circle_mask = (xx - y) ** 2 + (yy - x) ** 2 <= radius ** 2
        heatmap[circle_mask] += value

    # Load the dataset
    data = pd.read_csv(csv_path)

    # Initialize a set to track objects that have already been processed
    processed_objects = set()

    # Filter out rows where 'Object ID' is a merge product (contains parentheses)
    filtered_data = data[~data['label'].str.contains(r'\(.*\)', regex=True)]

    # Create an empty heatmap using numpy for fast operations
    heatmap = np.zeros((512, 512), dtype=np.float32)

    # Group data by Frame Number to process each frame independently
    grouped_by_frame = filtered_data.groupby('frame_number')

    # Process each frame group
    for frame_number, frame_group in grouped_by_frame:
        for _, row in frame_group.iterrows():
            object_id = row['label']
            # Extract the main object ID (e.g., 1_frame0) without merge indicators
            base_object_id = object_id.split('(')[0]

            # Check if this object has already been processed
            if base_object_id not in processed_objects:
                # If it's a new object (first instance), process it
                x, y = int(row['centroid_x']), int(row['centroid_y'])
                draw_circle(heatmap, x, y)

                # Mark the object as processed
                processed_objects.add(base_object_id)

    # Normalize the heatmap for better visualization
    heatmap_normalized = heatmap / np.max(heatmap)

    # Plot the heatmap
    plt.figure(figsize=(12, 10))

    # Create the desired colormap
    cmap = mcolors.LinearSegmentedColormap.from_list("blue_red_yellow", ["blue", "cyan", "yellow", "red"])

    # Create the heatmap using seaborn's heatmap function
    ax = sns.heatmap(
        heatmap_normalized,
        cbar=True,
        cmap=cmap,
        cbar_kws={'label': 'Frequency'}
    )

    # Set title with the desired font size
    plt.title(f"Bubble Emerging Points Density Map for {case_study}", fontsize=28)

    # Remove x and y ticks
    plt.xticks([])
    plt.yticks([])

    # Adjust font size for colorbar ticks and label
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=28)  # Adjust colorbar tick labels
    cbar.set_label('Frequency', size=28)  # Adjust colorbar label size

    # Extract the directory from the CSV path and construct the heatmap save path
    csv_dir = os.path.dirname(csv_path)
    heatmap_save_path = os.path.join(analysis_save_path, f"Bubble Emerging Points Heatmap for {case_study}.png")

    # Save the heatmap to the same folder as the input CSV file
    plt.savefig(heatmap_save_path, bbox_inches='tight', dpi=300)
    print(f"Heatmap saved to {heatmap_save_path}")

    # Display the heatmap
    plt.show()

##### MERGING POINTS HEATMAPS ######

def generate_merging_heatmap(csv_path, case_study):
    """
    Generates a heatmap of merging events based on their centroids.

    Args:
        csv_path (str): Path to the CSV file containing tracking data.
        case_study (str): Case details associated with the dataset (e.g., "1500mV").
    """

    parent_directory = Path(csv_path).parent
    analysis_save_path = parent_directory / f"Analysis Plots for {case_study}"
    analysis_save_path.mkdir(parents=True, exist_ok=True)

    # Optimized function to draw circles using vectorized operations
    def draw_circle(heatmap, x, y, radius=15, value=0.4):
        """Draws a circle on the heatmap centered at (x, y)."""
        xx, yy = np.ogrid[:heatmap.shape[0], :heatmap.shape[1]]
        circle_mask = (xx - y) ** 2 + (yy - x) ** 2 <= radius ** 2
        heatmap[circle_mask] += value

    # Load the dataset
    data = pd.read_csv(csv_path)

    # Create an empty heatmap using numpy for fast operations
    heatmap = np.zeros((512, 512), dtype=np.float32)

    # Filter rows where 'label' indicates a merging event (contains parentheses)
    merging_data = data[data['label'].str.contains(r'\(.*\)', regex=True)]

    # Process unique merging events
    for label in merging_data['label'].unique():
        # Get the first occurrence of the merging event
        first_occurrence = merging_data[merging_data['label'] == label].nsmallest(1, 'frame_number').iloc[0]
        
        # Extract centroid coordinates
        x, y = int(first_occurrence['centroid_x']), int(first_occurrence['centroid_y'])
        draw_circle(heatmap, x, y)

    # Normalize the heatmap for better visualization
    heatmap_normalized = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap

    # Plot the heatmap
    plt.figure(figsize=(12, 10))

    # Create the desired colormap
    cmap = mcolors.LinearSegmentedColormap.from_list("blue_red_yellow", ["blue", "cyan", "yellow", "red"])

    # Create the heatmap using seaborn's heatmap function
    ax = sns.heatmap(
        heatmap_normalized,
        cbar=True,
        cmap=cmap,
        cbar_kws={'label': 'Frequency'}
    )

    # Set title with the desired font size
    plt.title(f"Bubble Merging Points Density Map for {case_study}", fontsize=28)

    # Remove x and y ticks
    plt.xticks([])
    plt.yticks([])

    # Adjust font size for colorbar ticks and label
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=28)  # Adjust colorbar tick labels
    cbar.set_label('Frequency', size=28)  # Adjust colorbar label size

    # Construct the heatmap save path
    mrg_heatmap_save_path = os.path.join(analysis_save_path, f"Bubble Merging Points Density Map for {case_study}.png")

    # Save the heatmap to the specified folder
    plt.savefig(mrg_heatmap_save_path, bbox_inches='tight', dpi=300)
    print(f"Heatmap saved to {mrg_heatmap_save_path}")

    # Display the heatmap
    plt.show()


### BUBBLE TRAJECTORIES ###

def analyze_bubble_trajectories_and_velocities(csv_path, case_study):
    """
    Analyze bubble trajectories and velocities, and generate trajectory plots with enhanced legend 
    for a given CSV file.

    Args:
        csv_path (str): Path to the CSV file containing bubble tracking data.
        case_study (str): Case details associated with the dataset (e.g., "1500mV").

    Returns:
        dict: A dictionary containing calculated metrics for singular and merged bubbles.
    """
    # Load the dataset
    data = pd.read_csv(csv_path)

    # Create a folder for analysis plots
    save_path = Path(csv_path).parent / f"Analysis Plots for {case_study}"
    save_path.mkdir(parents=True, exist_ok=True)

    # Separate singular and merged objects based on the label format
    data['is_merged'] = data['label'].str.contains(r'\(.*\)')
    singular_data = data[~data['is_merged']]
    merged_data = data[data['is_merged']]

    # Initialize lists to store metrics for singular and merged bubbles
    singular_total_distances, singular_net_displacements, singular_velocities = [], [], []
    merged_total_distances, merged_net_displacements, merged_velocities = [], [], []

    # Function to compute the Euclidean distance between consecutive points
    def calculate_total_distance(x_coords, y_coords):
        if len(x_coords) < 2:
            return 0
        distances = np.sqrt(np.diff(x_coords) ** 2 + np.diff(y_coords) ** 2)
        return np.sum(distances)

    # Function to compute the straight-line distance from start to end
    def calculate_net_displacement(x_coords, y_coords):
        if len(x_coords) < 2:
            return 0
        return np.sqrt((x_coords[-1] - x_coords[0]) ** 2 + (y_coords[-1] - y_coords[0]) ** 2)

    # Function to calculate average velocity
    def calculate_velocity(net_displacement, start_frame, end_frame):
        timespan = end_frame - start_frame
        return net_displacement / timespan if timespan > 0 else 0  # Prevent division by zero

    # Calculate metrics for singular bubbles
    for object_id, group in singular_data.groupby('label'):
        x_coords, y_coords = group['centroid_x'].values, group['centroid_y'].values
        start_frame, end_frame = group['frame_number'].values[0], group['frame_number'].values[-1]

        total_distance = calculate_total_distance(x_coords, y_coords)
        net_displacement = calculate_net_displacement(x_coords, y_coords)
        velocity = calculate_velocity(net_displacement, start_frame, end_frame)

        singular_total_distances.append(total_distance)
        singular_net_displacements.append(net_displacement)
        singular_velocities.append(velocity)

    # Calculate metrics for merged bubbles
    for object_id, group in merged_data.groupby('label'):
        x_coords, y_coords = group['centroid_x'].values, group['centroid_y'].values
        start_frame, end_frame = group['frame_number'].values[0], group['frame_number'].values[-1]

        total_distance = calculate_total_distance(x_coords, y_coords)
        net_displacement = calculate_net_displacement(x_coords, y_coords)
        velocity = calculate_velocity(net_displacement, start_frame, end_frame)

        merged_total_distances.append(total_distance)
        merged_net_displacements.append(net_displacement)
        merged_velocities.append(velocity)

    # Calculate averages and standard deviations
    def calculate_stats(values):
        return np.mean(values) if values else 0, np.std(values) if values else 0

    singular_avg_dist, singular_std_dist = calculate_stats(singular_total_distances)
    singular_avg_disp, singular_std_disp = calculate_stats(singular_net_displacements)
    singular_avg_vel, singular_std_vel = calculate_stats(singular_velocities)

    merged_avg_dist, merged_std_dist = calculate_stats(merged_total_distances)
    merged_avg_disp, merged_std_disp = calculate_stats(merged_net_displacements)
    merged_avg_vel, merged_std_vel = calculate_stats(merged_velocities)

    # Create the trajectory plot
    plt.figure(figsize=(15, 18))
    plt.title(f"Bubble Trajectories for {case_study}", fontsize=28)
    plt.xlabel('Centroid X', fontsize=28)
    plt.ylabel('Centroid Y', fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)

    # Plot singular trajectories
    for _, group in singular_data.groupby('label'):
        x_coords, y_coords = group['centroid_x'].values, group['centroid_y'].values
        plt.plot(x_coords, y_coords, linestyle='-', linewidth=0.8, color='red')
        plt.plot(x_coords[0], y_coords[0], marker='o', markersize=5, color='red')
        plt.plot(x_coords[-1], y_coords[-1], marker='o', markersize=5, color='red')

    # Plot merged trajectories
    for _, group in merged_data.groupby('label'):
        x_coords, y_coords = group['centroid_x'].values, group['centroid_y'].values
        plt.plot(x_coords, y_coords, linestyle='-', linewidth=0.8, color='lime')
        plt.plot(x_coords[0], y_coords[0], marker='o', markersize=5, color='lime')
        plt.plot(x_coords[-1], y_coords[-1], marker='o', markersize=5, color='lime')

    # Enhanced legend
    plt.plot([], [], color='red', label=(
        f'Singular Bubbles\n'
        f'Avg Total Distance: {singular_avg_dist:.2f} px (±{singular_std_dist:.2f})\n'
        f'Avg Displacement: {singular_avg_disp:.2f} px (±{singular_std_disp:.2f})\n'
        f'Avg Velocity: {singular_avg_vel:.2f} px/frame (±{singular_std_vel:.2f})'))
    plt.plot([], [], color='lime', label=(
        f'Merged Bubbles\n'
        f'Avg Total Distance: {merged_avg_dist:.2f} px (±{merged_std_dist:.2f})\n'
        f'Avg Displacement: {merged_avg_disp:.2f} px (±{merged_std_disp:.2f})\n'
        f'Avg Velocity: {merged_avg_vel:.2f} px/frame (±{merged_std_vel:.2f})'))
    
    # Position the legend outside the plot
    plt.legend(loc="lower right", bbox_to_anchor=(1.45, 0), fontsize=24, title="Trajectory Legend", title_fontsize=28)


    # Save the plot
    plot_save_path = save_path / f"trajectory_{case_study}_with_metrics.png"
    plt.savefig(plot_save_path, bbox_inches='tight', dpi=300)
    plt.show()

    print(f"Trajectory plot saved to: {plot_save_path}")

    # Return metrics for further use
    metrics = {
        "Singular Avg Total Distance": singular_avg_dist,
        "Singular Avg Displacement": singular_avg_disp,
        "Singular Avg Velocity": singular_avg_vel,
        "Merged Avg Total Distance": merged_avg_dist,
        "Merged Avg Displacement": merged_avg_disp,
        "Merged Avg Velocity": merged_avg_vel
    }
    return metrics




#### BUBBLE METRIC ANALYSIS ###

def bubble_metrics_analysis(csv_path, movement_threshold, case_study):
    """
    Perform a comprehensive analysis of bubble metrics from a single CSV file.

    Args:
        csv_path (str): Path to the CSV file.
        movement_threshold (float): Threshold for net movement (in pixels).
        case_study (str): Case details associated with the dataset.
    
    Outputs:
        - Summary CSV with metrics in "Metric" and "Value" columns.
    """
    # Define save path
    parent_directory = Path(csv_path).parent
    analysis_save_path = parent_directory / f"Analysis Plots for {case_study}"
    analysis_save_path.mkdir(parents=True, exist_ok=True)

    # Load the dataset
    df = pd.read_csv(csv_path)

    # Initialize a list for summary metrics
    summary = []

    # Helper function: Filter objects without parentheses in label
    def filter_objects(df):
        return df[~df['label'].str.contains(r'\(.*\)', regex=True)].copy()

    # Function to compute total distance, net displacement, and velocity
    def calculate_metrics(group):
        x_coords, y_coords = group['centroid_x'].values, group['centroid_y'].values
        start_frame, end_frame = group['frame_number'].values[0], group['frame_number'].values[-1]
        if len(x_coords) < 2:
            return 0, 0, 0
        # Total distance
        total_distance = np.sum(np.sqrt(np.diff(x_coords) ** 2 + np.diff(y_coords) ** 2))
        # Net displacement
        net_displacement = np.sqrt((x_coords[-1] - x_coords[0]) ** 2 + (y_coords[-1] - y_coords[0]) ** 2)
        # Average velocity
        velocity = net_displacement / (end_frame - start_frame) if end_frame > start_frame else 0
        return total_distance, net_displacement, velocity

    # Separate singular and merged bubbles
    df['is_merged'] = df['label'].str.contains(r'\(.*\)')
    singular_data = df[~df['is_merged']]
    merged_data = df[df['is_merged']]

    # Calculate metrics for singular and merged bubbles
    singular_metrics = {'total_distances': [], 'net_displacements': [], 'velocities': []}
    merged_metrics = {'total_distances': [], 'net_displacements': [], 'velocities': []}

    for _, group in singular_data.groupby('label'):
        total_distance, net_displacement, velocity = calculate_metrics(group)
        singular_metrics['total_distances'].append(total_distance)
        singular_metrics['net_displacements'].append(net_displacement)
        singular_metrics['velocities'].append(velocity)

    for _, group in merged_data.groupby('label'):
        total_distance, net_displacement, velocity = calculate_metrics(group)
        merged_metrics['total_distances'].append(total_distance)
        merged_metrics['net_displacements'].append(net_displacement)
        merged_metrics['velocities'].append(velocity)

    # Calculate averages for singular and merged bubbles
    def calculate_averages(metrics_dict):
        return {key: np.mean(values) if values else None for key, values in metrics_dict.items()}

    singular_avg = calculate_averages(singular_metrics)
    merged_avg = calculate_averages(merged_metrics)

    # Add singular and merged metrics to summary
    summary.append({"Metric": "Singular Avg Total Distance (Pixels)", "Value": singular_avg['total_distances']})
    summary.append({"Metric": "Singular Avg Displacement (Pixels)", "Value": singular_avg['net_displacements']})
    summary.append({"Metric": "Singular Avg Velocity (Pixels/Frame)", "Value": singular_avg['velocities']})
    summary.append({"Metric": "Merged Avg Total Distance (Pixels)", "Value": merged_avg['total_distances']})
    summary.append({"Metric": "Merged Avg Displacement (Pixels)", "Value": merged_avg['net_displacements']})
    summary.append({"Metric": "Merged Avg Velocity (Pixels/Frame)", "Value": merged_avg['velocities']})

    # 1. Average New Bubble Size
    df_filtered = filter_objects(df)
    avg_new_bubble_size = df_filtered['area'].mean()
    summary.append({"Metric": "Average New Bubble Size (All Bubbles, Pixels)", "Value": avg_new_bubble_size})

    # 2. Average Initial Bubble Size for Moving Bubbles
    initial_sizes = []
    for label, group in df_filtered.groupby('label'):
        group = group.sort_values('frame_number')
        start_x, start_y = group.iloc[0][['centroid_x', 'centroid_y']]
        start_area = group.iloc[0]['area']
        for i in range(1, len(group)):
            end_x, end_y = group.iloc[i][['centroid_x', 'centroid_y']]
            net_movement = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
            if net_movement > movement_threshold:
                initial_sizes.append(start_area)
                break
    avg_initial_size = np.mean(initial_sizes) if initial_sizes else None
    summary.append({"Metric": "Average Initial Bubble Size for Moving Bubbles (Pixels)", "Value": avg_initial_size})

    # 3. Average Final Bubble Size for Moving Bubbles
    final_sizes = []
    for label, group in df_filtered.groupby('label'):
        group = group.sort_values('frame_number')
        start_x, start_y = group.iloc[0][['centroid_x', 'centroid_y']]
        for i in range(1, len(group)):
            end_x, end_y = group.iloc[i][['centroid_x', 'centroid_y']]
            net_movement = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
            if net_movement > movement_threshold:
                final_sizes.append(group.iloc[i]['area'])
                break
    avg_final_size = np.mean(final_sizes) if final_sizes else None
    summary.append({"Metric": "Average Final Bubble Size of Moving Bubbles (Pixels)", "Value": avg_final_size})

    # 4. Growth Rate Before Movement
    growth_rates = []
    for label, group in df_filtered.groupby('label'):
        group = group.sort_values('frame_number')
        start_x, start_y = group.iloc[0][['centroid_x', 'centroid_y']]
        start_area = group.iloc[0]['area']
        for i in range(1, len(group)):
            end_x, end_y = group.iloc[i][['centroid_x', 'centroid_y']]
            end_area = group.iloc[i]['area']
            net_movement = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
            if net_movement > movement_threshold:
                growth_rate = ((end_area / start_area) - 1) * 100
                growth_rates.append(growth_rate)
                break
    avg_growth_rate = np.mean(growth_rates) if growth_rates else None
    summary.append({"Metric": "Average Growth Rate of Moving Bubbles (%)", "Value": avg_growth_rate})

    # 5. Average Displacement
    start_points = df_filtered.groupby('label').first()[['centroid_x', 'centroid_y']]
    end_points = df_filtered.groupby('label').last()[['centroid_x', 'centroid_y']]
    displacements = np.sqrt(
        (end_points['centroid_x'] - start_points['centroid_x']) ** 2 +
        (end_points['centroid_y'] - start_points['centroid_y']) ** 2
    )
    avg_displacement = displacements[displacements > movement_threshold].mean() if not displacements.empty else None
    summary.append({"Metric": "Average Displacement of Moving Bubbles (Pixels)", "Value": avg_displacement})

    # Convert summary list to DataFrame and save to CSV
    summary_df = pd.DataFrame(summary)
    summary_csv_path = analysis_save_path / f"bubble_metrics_summary_{case_study}.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Summary saved to {summary_csv_path}")

    print("Analysis completed. Metrics calculated and saved.")

