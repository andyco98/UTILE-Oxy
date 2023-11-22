import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import re
import csv

# # Create a directory for the output images
# output_folder = "./Output_Images"
# os.makedirs(output_folder, exist_ok=True)

# Extract ROIs from all masks in a folder
experiment = "NameYourExperiment"  #GIve a name or ID to your experiment
mask_folder = "/p/home/jusers/colliard1/juwels/fzj-mac/Andre/Bubbles/UNet/RodenFullDataset/18mv_pred/"
os.makedirs(mask_folder, exist_ok=True)  # Ensure mask folder exists


# Function to calculate area and equivalent diameter
def calculate_area_and_diameter(img):
    area = cv2.countNonZero(img)
    equivalent_diameter = np.sqrt(4 * area / np.pi)
    return area, equivalent_diameter

 

# Function to calculate aspect ratio
def calculate_aspect_ratio(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    _, _, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    return aspect_ratio

 

def calculate_solidity(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    contour_area = cv2.contourArea(contour)
    if hull_area == 0:
        return 0 # Or return any other special value to indicate undefined solidity
    else:
        solidity = float(contour_area) / hull_area
        return solidity

 

def calculate_orientation(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)  # Select the contour with the greatest area
    if len(contour) < 5:
        return 0  # Or any other value to indicate undefined orientation
    else:
        (_, _), (_, _), angle = cv2.fitEllipse(contour)
        return angle

 

# Function to calculate extent
def calculate_extent(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h
    extent = float(contour_area) / rect_area
    return extent

 

# Function to calculate perimeter
def calculate_perimeter(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    return perimeter

 
def calculate_roundness(img):
    # Find contours
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Get the largest contour
    contour = max(contours, key=cv2.contourArea)
    
    # Find the area of the contour
    area = cv2.contourArea(contour)
    
    # Get the minimum enclosing circle
    (x,y), radius = cv2.minEnclosingCircle(contour)
    min_circle_area = np.pi * radius * radius

    if min_circle_area == 0:
        return 0  # to prevent division by zero

    # Calculate roundness
    roundness = area / min_circle_area
    
    return roundness

 

def draw_metrics(img, metrics, position=(50, 50), font=cv2.FONT_HERSHEY_SIMPLEX, 
                 font_scale=1, color=(255, 255, 255), thickness=2):
    y = position[1]
    for key, value in metrics.items():
        if value is None:  # If the value is None, print N/A
            text = f"{key}: N/A"
        else:
            text = f"{key}: {value:.2f}"
        cv2.putText(img, text, (position[0], y), font, font_scale, color, thickness)
        y += 30  # Move down for next line

 



params = ["area","diameter","aspect_ratio","solidity","orientation","extent","perimeter","roundness"]



def extract_number_from_filename(filename):
    match = re.search(r'pred_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0  # Default value


def individual_bubble_analysis(mask_folder):
    #Time resolved mean values for each frame

    exp_mean_area = []
    exp_mean_diameter = []
    exp_mean_aspect_ratio = []
    exp_mean_solidity = []
    exp_mean_orientation = []
    exp_mean_extent = []
    exp_mean_perimeter = []
    exp_mean_roundness = []

    #All values for each bubble over the whole experiment
    exp_total_area = []
    exp_total_diameter = []
    exp_total_aspect_ratio = []
    exp_total_solidity = []
    exp_total_orientation = []
    exp_total_extent = []
    exp_total_perimeter = []
    exp_total_roundness = []
    mask_files_sorted = sorted(os.listdir(mask_folder), key=extract_number_from_filename)


    # Iterate over all mask files in the directory
    for mask_file in mask_files_sorted:
        mask_file_path = os.path.join(mask_folder, mask_file)
        mask = np.array(Image.open(mask_file_path))  # Load mask as grayscale

        frame_total_area = []
        frame_total_diameter = []
        frame_total_aspect_ratio = []
        frame_total_solidity = []
        frame_total_orientation = []
        frame_total_extent = []
        frame_total_perimeter = []
        frame_total_roundness = []



        # Find contours (i.e., bubbles) in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    

        # Iterate over each contour/bubble
        for i, contour in enumerate(contours):
            # Create a blank image to draw the current contour/bubble on
            roi = np.zeros_like(mask)
            cv2.drawContours(roi, [contour], -1, (255), thickness=cv2.FILLED)

            # Calculate metrics for the current ROI
            area, diameter = calculate_area_and_diameter(roi)
            aspect_ratio = calculate_aspect_ratio(roi)
            solidity = calculate_solidity(roi)
            orientation = calculate_orientation(roi)
            extent = calculate_extent(roi)
            perimeter = calculate_perimeter(roi)
            roundness = calculate_roundness(roi)

            frame_total_area.append(area)
            frame_total_diameter.append(diameter)
            frame_total_aspect_ratio.append(aspect_ratio)
            frame_total_solidity.append(solidity)
            frame_total_orientation.append(orientation)
            frame_total_extent.append(extent)
            frame_total_perimeter.append(perimeter)
            frame_total_roundness.append(roundness)

            exp_total_area.append(area)
            exp_total_diameter.append(diameter)
            exp_total_aspect_ratio.append(aspect_ratio)
            exp_total_solidity.append(solidity)
            exp_total_orientation.append(orientation)
            exp_total_extent.append(extent)
            exp_total_perimeter.append(perimeter)
            exp_total_roundness.append(roundness)

            # Store metrics in a dictionary
            metrics = {
                'Area': area,
                'Diameter': diameter,
                'Aspect Ratio': aspect_ratio,
                'Solidity': solidity,
                'Orientation': orientation,
                'Extent': extent,
                'Perimeter': perimeter,
                'Roundness': roundness
            }
        
        exp_mean_area.append(np.mean(frame_total_area))
        exp_mean_diameter.append(np.mean(frame_total_diameter))
        exp_mean_aspect_ratio.append(np.mean(frame_total_aspect_ratio))
        exp_mean_solidity.append(np.mean(frame_total_solidity))
        exp_mean_orientation.append(np.mean(frame_total_orientation))
        exp_mean_extent.append(np.mean(frame_total_extent))
        exp_mean_perimeter.append(np.mean(frame_total_perimeter))
        exp_mean_roundness.append(np.mean(frame_total_roundness))



        
        
    headers = ["exp_mean_area","exp_mean_diameter","exp_mean_aspect_ratio","exp_mean_solidity", "exp_mean_orientation","exp_mean_extent","exp_mean_perimeter","exp_mean_roundness"]


    data = list(zip(exp_mean_area, exp_mean_diameter, exp_mean_aspect_ratio, exp_mean_solidity, exp_mean_orientation, exp_mean_extent, exp_mean_perimeter, exp_mean_roundness))


    # Specify the filename
    filename = f"./output_csv.csv"

    # Write to CSV
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Write the headers
        csvwriter.writerow(headers)
        
        # Write the data
        csvwriter.writerows(data)

    fig, ax = plt.subplots(4,2, figsize=(10,8))

    # Adding the curves to the first subplot
    ax[0,0].plot(exp_mean_area, label='Area')

    ax[0,1].plot(exp_mean_diameter, label='Diameter')

    ax[1,0].plot(exp_mean_aspect_ratio, label='Aspect Ratio')

    ax[1,1].plot(exp_mean_solidity, label='Solidity')

    ax[2,0].plot(exp_mean_orientation, label='Orientation')

    ax[2,1].plot(exp_mean_extent, label='Extent')

    ax[3,0].plot(exp_mean_perimeter, label='Perimeter')

    ax[3,1].plot(exp_mean_roundness, label='Roundness')

    #ax[0,0].legend()
    ax[0,0].set_title('Area')
    ax[0,1].set_title('Diameter')
    ax[1,0].set_title('Aspect Ratio')
    ax[1,1].set_title('Solidity')
    ax[2,0].set_title('Orientation')
    ax[2,1].set_title('Extent')
    ax[3,0].set_title('Perimeter')
    ax[3,1].set_title('Roundness')
    plt.tight_layout()
    plt.savefig(f"./individual_bubble_analysis.png")
    plt.show()

