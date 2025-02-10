import os
from PIL import Image
import numpy as np
import tifffile
from scipy.ndimage import label, generate_binary_structure
from skimage.color import label2rgb
import cv2
import pandas as pd
import time
from collections import defaultdict
from pathlib import Path
import re
import matplotlib.pyplot as plt
from skimage import io
from tracking_standalone import*

global case_study

# Setting up the universal output
def setup_directories(input_folder):
    """
    Sets up the parent directory based on the mask_folder path.
    """
    if not os.path.isdir(input_folder):
        raise ValueError(f"Provided mask folder does not exist: {input_folder}")
    return os.path.dirname(input_folder.rstrip('/'))  # Ensure no trailing slash

# Creating the binary TIFF stack from the mask folder
def create_binary_tiff_stack(input_dir, case_study, chunk_size=500):
    """
    Creates a binary TIFF stack from PNG masks for tracking purposes.
    """
    output_dir=setup_directories(input_dir)
    output_path_binary_tiff = os.path.join(output_dir, f"binary_stack_{case_study}.tiff")
    png_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])

    if not png_files:
        raise ValueError(f"No PNG files found in directory: {input_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(output_path_binary_tiff):
        os.remove(output_path_binary_tiff)

    for i in range(0, len(png_files), chunk_size):
        chunk_files = png_files[i:i + chunk_size]
        images = [cv2.imread(os.path.join(input_dir, file), cv2.IMREAD_GRAYSCALE) for file in chunk_files]
        stack = np.stack(images, axis=0)
        tifffile.imwrite(output_path_binary_tiff, stack, bigtiff=True, append=i > 0)

    print(f"Binary TIFF stack created: {output_path_binary_tiff}")
    return output_path_binary_tiff

# 3D Volume Cleaning and Labeling
def clean_volume(volume, target_class=255, min_size=25):
    """
    Cleans the volume by removing isolated pixels and small objects.
    """
    struct = generate_binary_structure(3, 3)
    binary_target = (volume == target_class)
    labeled_array, num_features = label(binary_target, structure=struct)

    sizes = np.bincount(labeled_array.ravel())
    removal_mask = sizes[labeled_array] <= min_size
    volume[removal_mask] = 0

    return volume

def label_bubbles(volume, bubble_class=255):
    """
    Labels bubbles in the volume.
    """
    binary_bubbles = volume == bubble_class
    struct = generate_binary_structure(3, 3)
    labeled_volume, num_features = label(binary_bubbles, structure=struct)
    return labeled_volume, num_features

def colorize_labels(labeled_volume):
    """
    Applies random colors to labeled regions for visualization.
    """
    max_label = labeled_volume.max()
    colors = np.random.randint(0, 255, size=(max_label + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Keep the background black
    return colors[labeled_volume]

def process_and_save_labeled_stack(binary_stack_path, case_study):
    """
    Processes a binary TIFF stack to clean and label it, then saves the labeled stack.
    """
    output_dir = setup_directories(os.path.dirname(binary_stack_path))
    
    output_path_labeled_tiff = os.path.join(output_dir, f"labeled_cleaned_stack_{case_study}.tiff")
    
    with tifffile.TiffFile(binary_stack_path) as tif:
        volume = np.stack([page.asarray() for page in tif.pages])

    cleaned_volume = clean_volume(volume)
    labeled_volume, num_features = label_bubbles(cleaned_volume)
    colorized_volume = colorize_labels(labeled_volume)

    tifffile.imwrite(output_path_labeled_tiff, colorized_volume)
    print(f"Labeled stack saved: {output_path_labeled_tiff}")
    return output_path_labeled_tiff

def tracking_bubbles(input_mask_folder, case_study):
    print("Initializing...")
    output_dir=setup_directories(input_mask_folder)
    step1=create_binary_tiff_stack(input_mask_folder, case_study, chunk_size=500)
    step2=process_and_save_labeled_stack(step1, case_study)
    global output_tiff_stack_path
    global combined_csv_path
    output_tiff_stack_path = os.path.join(output_dir, f"{case_study}_tracked.tiff")
    combined_csv_path = os.path.join(output_dir, f"{case_study}_tracking_datas.csv")
    step3=tracking_standalone(step2, output_tiff_stack_path, combined_csv_path)
    os.remove(step1)
    os.remove(step2)
    print(f"Intermediate files are cleaned")
    return combined_csv_path, output_tiff_stack_path
    
    




