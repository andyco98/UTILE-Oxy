    
import cv2
import numpy as np
import pandas as pd
import re
import time
from collections import defaultdict
import tifffile
import matplotlib.pyplot as plt
from skimage import io

def tracking_standalone(full_path, output_tiff_stack_path, combined_csv_path):

    # ----------------------------
    # LABEL STORAGE AND GLOBALS
    # ----------------------------
    global label_data
    global global_instance_counter
    global frame_creation
    
    labels_data = []
    global_instance_counter = 0
    frame_creation = {}
    
    # Initialize global sets for active and inactive labels
    inactive_labels = set()
    active_labels = set()
    
    # Batch size for processing large TIFF stacks
    BATCH_SIZE = 500  # Adjust this based on your system's memory capacity 500 is reasonably fast
    
    # ----------------------------
    # LABEL GENERATION FUNCTIONS
    # ----------------------------
    def generate_label(instance_counter, frame_number):
        """Generate a unique label based on instance counter and frame number."""
        return f"{instance_counter}_frame{frame_number}"
    
    def generate_merge_label(instance_counter, parent_labels, frame_number):
        """
        Generate a unique label for merged objects following the pattern:
        "{new_instance}_frame{frame_number}_({parent1}+{parent2}+...)".
        Example: "7_frame2_(3+4)"
        """
        parents = "+".join(parent_labels)
        return f"{instance_counter}_frame{frame_number}_({parents})"
    
    # ----------------------------
    # IMAGE PROCESSING FUNCTIONS
    # ----------------------------
    def preprocess_image_otsu(frame):
        """Convert frame to grayscale and apply Otsu's thresholding."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    def detect_contours(binary_image):
        """Detect external contours in the binary image."""
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def calculate_metrics(img, contour):
        """Calculate various metrics for a given contour."""
        area = cv2.contourArea(contour)
        equivalent_diameter = np.sqrt(4 * area / np.pi)
        _, _, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h != 0 else 0
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        orientation = cv2.fitEllipse(contour)[-1] if len(contour) >= 5 else 0
        rect_area = w * h
        extent = float(area) / rect_area if rect_area > 0 else 0
        perimeter = cv2.arcLength(contour, True)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        min_circle_area = np.pi * radius * radius
        roundness = area / min_circle_area if min_circle_area > 0 else 0
        return {
            "area": area,
            "diameter": equivalent_diameter,
            "aspect_ratio": aspect_ratio,
            "solidity": solidity,
            "orientation": orientation,
            "extent": extent,
            "perimeter": perimeter,
            "roundness": roundness
        }
    
    # ----------------------------
    # SPLIT HANDLING FUNCTIONS
    # ----------------------------
    def average_split_centroids(detected_splits):
        """Average the centroids of split objects to smooth their positions."""
        for label, data in detected_splits.items():
            centroids = data["centroids"]
            count = data["count"]
            avg_cX = int(sum(x for x, y in centroids) / count)
            avg_cY = int(sum(y for x, y in centroids) / count)
            
            # Update labels_data with averaged centroid for all occurrences of the split label
            for entry in labels_data:
                if entry["label"] == label:
                    entry["centroid_x"] = avg_cX
                    entry["centroid_y"] = avg_cY
    
    # ----------------------------
    # CONTOUR PROCESSING FUNCTION
    # ----------------------------
    def process_contour(contour, frame_lab, previous_frame_objects, frame_number, detected_splits, current_frame_label_usage, frame_creation, metrics_dict):
        """
        Process a single contour to assign labels, handle splits, and calculate metrics.
        """
        global global_instance_counter
        mask = np.zeros(frame_lab.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    
        overlapping_labels = []
        for prev_label, prev_masks in previous_frame_objects.items():
            for prev_mask in prev_masks:
                if np.logical_and(mask, prev_mask).any():
                    overlapping_labels.append(prev_label)
                    break  # Stop after finding an overlap with this label
    
        generated_label = None
        if len(overlapping_labels) == 1:
            prev_label = overlapping_labels[0]
            is_split_product = any(
                split_label for split_label in detected_splits.keys() 
                if split_label == prev_label or 
                   (isinstance(prev_label, str) and prev_label.startswith(split_label.split('_frame')[0]))
            )
    
            if current_frame_label_usage[prev_label] > 0:
                if prev_label not in detected_splits:
                    detected_splits[prev_label] = {
                        "centroids": [(cX, cY)],
                        "count": 1,
                        "frame_detected": frame_number
                    }
                    inactive_labels.add(prev_label)
                    active_labels.discard(prev_label)
                else:
                    detected_splits[prev_label]["centroids"].append((cX, cY))
                    detected_splits[prev_label]["count"] += 1
    
                generated_label = prev_label
                frame_creation[generated_label] = frame_creation.get(prev_label, frame_number)
            elif is_split_product:
                generated_label = prev_label
                frame_creation[generated_label] = frame_creation.get(prev_label, frame_number)
            else:
                generated_label = prev_label
    
            current_frame_label_usage[prev_label] += 1
        elif len(overlapping_labels) > 1:
            global_instance_counter += 1
            parent_labels = [label.split('_')[0] for label in overlapping_labels]
            generated_label = generate_merge_label(global_instance_counter, parent_labels, frame_number)
            frame_creation[generated_label] = frame_number
            current_frame_label_usage[generated_label] += 1
        else:
            found_nearby_split = False
            for split_label, split_data in detected_splits.items():
                if frame_number - split_data.get("frame_detected", 0) <= 2:
                    for split_centroid in split_data["centroids"]:
                        distance = np.sqrt((cX - split_centroid[0])**2 + (cY - split_centroid[1])**2)
                        if distance < 50:
                            generated_label = split_label
                            found_nearby_split = True
                            break
                if found_nearby_split:
                    break
    
            if not found_nearby_split:
                global_instance_counter += 1
                generated_label = generate_label(global_instance_counter, frame_number)
                frame_creation[generated_label] = frame_number
    
            current_frame_label_usage[generated_label] += 1
    
        if generated_label not in frame_creation:
            frame_creation[generated_label] = frame_number
    
        labels_data.append({
            "label": generated_label,
            "frame_number": frame_number,
            "centroid_x": cX,
            "centroid_y": cY
        })
        return {
            "label": generated_label,
            "mask": mask,
            "centroid": (cY, cX),
            "contour": contour,
            "frame_number": frame_number
        }
    
    # ----------------------------
    # ANNOTATION FUNCTION
    # ----------------------------
    def draw_and_annotate_frame(frame, contour_labels):
        """Draw contours and annotations (labels and centroids) on the frame."""
        for result in contour_labels:
            generated_label = result["label"]
            roi = np.zeros_like(frame[:, :, 0])
            cv2.drawContours(roi, [result["contour"]], -1, 255, -1)
            cv2.circle(frame, (result["centroid"][1], result["centroid"][0]), 2, (255, 0, 0), -1)
            cv2.putText(frame, f"{generated_label}", (result["centroid"][1] - 50, result["centroid"][0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.drawContours(frame, [result["contour"]], -1, (0, 0, 255), 1)
        return frame
    
    # ----------------------------
    # FRAME PROCESSING FUNCTION
    # ----------------------------
    def process_frame(frame, previous_frame_objects, frame_number, metrics_dict):
        """
        Process a single frame: detect contours, assign labels, calculate metrics, and annotate.
        """
        global frame_creation
        global global_instance_counter
    
        if len(frame.shape) == 2 or frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        if frame.size == 0:
            return frame, previous_frame_objects
    
        new_frame_objects = defaultdict(list)
        current_frame_label_usage = defaultdict(int)
        detected_splits = {}
        contour_labels = []
    
        binary_image = preprocess_image_otsu(frame)
        contours = detect_contours(binary_image)
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    
        for contour in contours:
            result = process_contour(
                contour, frame_lab, previous_frame_objects, frame_number,
                detected_splits, current_frame_label_usage, frame_creation, metrics_dict
            )
            if result:
                contour_labels.append(result)
                label = result["label"]
                metrics = calculate_metrics(frame, result["contour"])
                for metric, value in metrics.items():
                    metrics_dict[frame_number][label][metric] = value
    
        average_split_centroids(detected_splits)
    
        frame = draw_and_annotate_frame(frame, contour_labels)
    
        for result in contour_labels:
            generated_label = result["label"]
            new_frame_objects[generated_label].append(result["mask"])
    
        previous_frame_objects = new_frame_objects
    
        return frame, previous_frame_objects
    
    # ----------------------------
    # MAIN PROCESSING LOOP
    # ----------------------------
    metrics_dict = defaultdict(lambda: defaultdict(dict))
    previous_frame_objects = defaultdict(list)
    
    start_time = time.time()
    
    with tifffile.TiffFile(full_path) as tif:  # Replace with actual input path
        num_frames = len(tif.pages)
        print(f"Total frames to process: {num_frames}") ## just to confirm
    
        with tifffile.TiffWriter(output_tiff_stack_path, bigtiff=True) as tif_writer:
            for batch_start in range(0, num_frames, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, num_frames)
                print(f"Processing frames {batch_start} to {batch_end - 1}...") ##just to confirm
                tiff_stack = tif.asarray(key=range(batch_start, batch_end))
    
                processed_stack = []
    
                for idx, frame in enumerate(tiff_stack):
                    frame_number = batch_start + idx
                    processed_frame, previous_frame_objects = process_frame(
                        frame, previous_frame_objects, frame_number, metrics_dict
                    )
                    processed_stack.append(processed_frame)
    
                for processed_frame in processed_stack:
                    tif_writer.write(processed_frame.astype(np.uint8))
    
    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds.") 
    
    ##SAVE CSV ##
    
    # Convert labels_data to a dictionary for faster lookups ##NEW VERSION 
    labels_dict = {(item["label"], item["frame_number"]): item for item in labels_data}
    
    metrics_rows = []
    for frame_number, frame_data in metrics_dict.items():
        for label, label_data in frame_data.items():
            centroid_data = labels_dict.get((label, frame_number), {})
            row = {
                "frame_number": frame_number,
                "label": label,
                "centroid_x": centroid_data.get("centroid_x", 0),
                "centroid_y": centroid_data.get("centroid_y", 0)
            }
            row.update(label_data)
            metrics_rows.append(row)
    
    combined_df = pd.DataFrame(metrics_rows)
    combined_df.fillna(0, inplace=True)
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"Combined metrics and coordinates saved to {combined_csv_path}")
