import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import re
from scipy.stats import norm


def calculate_bubble_area(image_path):
    """
    Calculate the bubble area from a binary image.
    White pixels (value of 255) represent bubbles.
    """
    # Load the image as grayscale
    image = Image.open(image_path)
    image = np.array(image)

    # Calculate the bubble area (count of white pixels)
    bubble_area = np.sum(image == 255)
    total_pixels = image.size

    # Calculate the ratio
    bubble_ratio = bubble_area / total_pixels

    return bubble_ratio

# List of image files in the directory
def extract_number_from_filename(filename):
    match = re.search(r'pred_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0  # Default value


def time_resolved_BAD(directory_path):
    mask_files_sorted = sorted(os.listdir(directory_path), key=extract_number_from_filename)

    # Calculate the bubble area for each image
    areas = []

    for img in mask_files_sorted:
        img_path = directory_path + img
        #print(img_path)
        bubble_ratio = calculate_bubble_area(img_path)
        areas.append(bubble_ratio)

    # Manually compute the histogram using numpy
    hist_counts, bin_edges = np.histogram(areas, bins=30)

    # Compute bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Normalize the histogram counts to get percentages
    hist_percentages = 100 * hist_counts / np.sum(hist_counts)


    # # Plot the results
    # Calculate the mean and standard deviation of your areas
    mu, std = np.mean(areas), np.std(areas)

    # Plot the results
    fig, ax = plt.subplots(1,2, figsize=(12,5))

    ax[0].plot(areas)
    ax[0].set_xlabel('Frame number', fontsize = 18)
    ax[0].set_ylabel('Bubble ratio', fontsize = 18)
    ax[0].set_title('Time-resolved bubble ratio', fontsize = 18)
    ax[0].tick_params(axis='x', labelsize=16)
    ax[0].tick_params(axis='y', labelsize=16)

    ax[1].bar(bin_centers, hist_percentages, width=np.diff(bin_edges), edgecolor='black', alpha=0.7)
    ax[1].set_title('Bubble area distribution', fontsize = 18)
    ax[1].set_xlabel('Bubble ratio', fontsize = 18)
    ax[1].set_ylabel('Frequency [%]', fontsize = 18)
    ax[1].set_ylim(0, max(hist_percentages) + 5)  # Adding a bit of padding on top
    ax[1].tick_params(axis='x', labelsize=16)
    ax[1].tick_params(axis='y', labelsize=16)
    # Plot the Gaussian distribution
    x = np.linspace(min(areas), max(areas), 1000)
    pdf = norm.pdf(x, mu, std)
    ax[1].plot(x, pdf * np.sum(hist_percentages) * np.diff(bin_edges)[0], '-', color='red', label=f'Mean={mu:.2f}, SD={std:.2f}')
            
    ax[1].legend(loc='upper right', fontsize = 16)


    plt.tight_layout()
    plt.savefig("./time_resolved_BAD.png")
    plt.show()

