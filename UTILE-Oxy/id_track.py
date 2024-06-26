import cv2
import numpy as np
import os

def channel_recognition(full_img):
    img = cv2.imread(full_img, cv2.IMREAD_GRAYSCALE)
    w,h = img.shape
    quarter = w//4

    snippet = img[0:h, quarter:quarter+10]

    _, binary_snippet = cv2.threshold(snippet, 127, 255, cv2.THRESH_BINARY)

    cns, _ = cv2.findContours(binary_snippet, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    channel_positions = []

    for i in cns:
        x, y, w, h = cv2.boundingRect(i)
        start = y
        end = y+h

        channel_positions.append((start,end))

    channel_positions.sort()
    print(channel_positions)
    return channel_positions
    
def remove_outliers(distances, threshold=1.5):
    mean = np.mean(distances)
    std = np.std(distances)
    filtered_distances = [d for d in distances if abs(d - mean) <= threshold * std]
    return filtered_distances

def load_frames_from_folder(folder_path,channel):
    frames = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            frame = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
            if frame is None:
                print(f"Error: Could not read image {filename}.")
                continue
    
            cropped_image = frame[channel[0]-5:channel[1]+5, :]

            if frame is not None:
                _, binary_frame = cv2.threshold(cropped_image, 127, 255, cv2.THRESH_BINARY)
                frames.append(binary_frame)
    return frames

def calculate_centroids(contours):
    centroids = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            centroids.append((cX, cY))
    return centroids

def track_bubbles(frames):
    velocities = []
    bubble_tracks = {}
    bubble_id = 0

    prev_frame = frames[0]
    prev_contours, _ = cv2.findContours(prev_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    prev_centroids = calculate_centroids(prev_contours)

    for centroid in prev_centroids:
        bubble_tracks[bubble_id] = [centroid]
        bubble_id += 1

    for i in range(1, len(frames)):
        curr_frame = frames[i]
        curr_contours, _ = cv2.findContours(curr_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        curr_centroids = calculate_centroids(curr_contours)

        if len(prev_centroids) == 0 or len(curr_centroids) == 0:
            prev_centroids = curr_centroids
            continue

        # Match centroids from prev frame to current frame
        matched_pairs = []
        for prev_c in prev_centroids:
            distances = [np.linalg.norm(np.array(prev_c) - np.array(c)) for c in curr_centroids]
            if distances:
                min_index = np.argmin(distances)
                closest_centroid = curr_centroids[min_index]
                displacement = distances[min_index]
                matched_pairs.append((prev_c, closest_centroid, displacement))

        # Remove outliers
        displacements = [d for _, _, d in matched_pairs]
        filtered_displacements = remove_outliers(displacements)
        final_pairs = [pair for pair in matched_pairs if pair[2] in filtered_displacements]

        # Handle merging and splitting
        matched_curr_centroids = [pair[1] for pair in final_pairs]
        new_centroids = [c for c in curr_centroids if c not in matched_curr_centroids]

        # Update bubble_tracks with new positions
        for prev_c, curr_c, _ in final_pairs:
            for b_id, positions in bubble_tracks.items():
                if positions[-1] == prev_c:
                    bubble_tracks[b_id].append(curr_c)
                    break

        # Assign new IDs to unmatched centroids
        for new_c in new_centroids:
            bubble_tracks[bubble_id] = [new_c]
            bubble_id += 1

        prev_centroids = curr_centroids

        # Calculate average displacement (velocity) per frame
        if final_pairs:
            displacements = [pair[2] for pair in final_pairs]
            mean_velocity = np.mean(displacements)
            velocities.append(mean_velocity)
    mean_velocities = np.mean(velocities)

    return velocities, bubble_tracks, mean_velocities

def visualize_tracking(frames, bubble_tracks):
    color_map = {}
    color_idx = 0

    for b_id, positions in bubble_tracks.items():
        color_map[b_id] = (int((color_idx * 50) % 255), int((color_idx * 80) % 255), int((color_idx * 110) % 255))
        color_idx += 1

    for i, frame in enumerate(frames):
        vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        for b_id, positions in bubble_tracks.items():
            if i < len(positions):
                (x, y) = positions[i]
                cv2.circle(vis, (x, y), 5, color_map[b_id], -1)
                if i > 0:
                    cv2.line(vis, positions[i-1], (x, y), color_map[b_id], 2)

        cv2.imshow('Bubble Tracking', vis)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def channel_analysis(full_img, folder_path):

    channel_pos = channel_recognition(full_img)
    ch_num = 1
    ch_mean = []

    for channel in channel_pos:
        frames = load_frames_from_folder(folder_path, channel)
        velocities, bubble_tracks, mean = track_bubbles(frames)
        print(f'Channel {ch_num} mean velocity: {mean} px/frame ')
        ch_mean.append(mean)
        visualize_tracking(frames, bubble_tracks)
        ch_num += 1
    
    print(f'Mean cell velocity: {np.mean(ch_mean)} px/frame')

# Load frames from folder
#folder_path = f'./1Acm2_10mlmin/links_1'  # Change to your folder path
#full_img = './FullChannel512.png'

#channel_analysis(full_img, folder_path)

# Track bubbles and calculate velocities
#velocities, mean = track_bubbles(frames)

# Print velocities
#for i, v in enumerate(velocities):
    #print(f'Frame {i+1}: Average Bubble Velocity = {v:.2f} px/frame')
#print('mean velocity', mean)
# Visualize tracking
#visualize_tracking(frames)

#channel_recognition('./FullChannel512.png')