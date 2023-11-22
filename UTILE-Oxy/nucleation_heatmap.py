import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2

# Path to the directory containing the images
directory_path = "/p/home/jusers/colliard1/juwels/fzj-mac/Andre/Bubbles/UNet/RodenFullDataset/15mv_pred/"

def heatmap(directory_path):

    frames = []  

    # List of image files in the directory
    image_files = os.listdir(directory_path)

    for img in image_files:
        frame = Image.open(directory_path+img)
        frame = np.array(frame)
        _ , bin_frame = cv2.threshold(frame, 0,1, cv2.THRESH_BINARY)
        frames.append(bin_frame)

    print(len(frames), np.unique(frames[0]))

    # Assuming frames is a list of 2D numpy arrays with your semantic model predictions
    # List of your frames

    # Step 1: Initialize summation frame
    height, width = frames[0].shape
    summation_frame = np.zeros((height, width))

    # Step 2: Accumulate values
    for frame in frames:
        summation_frame += frame
        
    normalized_frame = summation_frame / np.max(summation_frame)

    # Visualize
    plt.imshow(normalized_frame, cmap='jet', interpolation='nearest')
    cbar = plt.colorbar()
    # Increase font size of colorbar ticks
    cbar.ax.tick_params(labelsize=18)  

    # Increase font size of axis ticks
    plt.xticks([])
    plt.yticks([])

    plt.title("Heatmap of Nucleation Points", fontsize=20)

    plt.savefig("./heatmap.png")
    plt.show()

