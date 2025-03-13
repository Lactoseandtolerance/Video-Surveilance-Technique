#!/usr/bin/env python3
# Video Surveillance Technique Enhancement

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.cluster import KMeans
import os
import time
import sys

class VideoSurveillanceSystem:
    def __init__(self, history=500, var_threshold=50, detect_shadows=True,
                 min_area=300, max_proposals=100):
        """Initialize the video surveillance system with parameters"""
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, 
            varThreshold=var_threshold, 
            detectShadows=detect_shadows
        )
        self.min_area = min_area
        self.max_proposals = max_proposals
        self.tracking_history = {}  # For object tracking
        self.next_object_id = 0
        
    def apply_background_subtraction(self, frame):
        """Apply background subtraction to identify moving objects"""
        # Apply background subtractor
        fg_mask = self.background_subtractor.apply(frame)
        
        # Apply morphological operations to improve mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Threshold to get binary mask (removing shadows)
        _, fg_mask_binary = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)
        
        return fg_mask, fg_mask_binary
    
    def extract_hog_features(self, frame):
        """Extract HOG (Histogram of Oriented Gradients) features"""
        # Ensure the input is grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Resize for consistent feature extraction
        resized = cv2.resize(gray_frame, (64, 128))
        
        # Extract HOG features and the visualization
        features, hog_image = hog(
            resized,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=True,
            feature_vector=True
        )
        
        # Normalize the visualization for display
        hog_image = (hog_image * 255).astype(np.uint8)
        return features, hog_image
    
    def detect_objects(self, frame, fg_mask_binary):
        """Detect objects using contour analysis"""
        objects = []
        contours, _ = cv2.findContours(fg_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            objects.append({
                'bbox': (x, y, w, h),
                'area': area,
                'centroid': (x + w//2, y + h//2)
            })
            
        return objects
    
    def track_objects(self, objects, frame):
        """Track objects across frames"""
        # If this is the first frame with objects
        if not self.tracking_history:
            for obj in objects:
                self.tracking_history[self.next_object_id] = {
                    'positions': [obj['centroid']],
                    'bbox': obj['bbox'],
                    'frames_tracked': 1,
                    'last_seen': 0
                }
                obj['id'] = self.next_object_id
                self.next_object_id += 1
            return objects
        
        # Match current objects with existing tracks
        current_frame_ids = set()
        max_distance = 50  # Maximum pixel distance for the same object
        
        for obj in objects:
            best_match = None
            min_dist = float('inf')
            
            for obj_id, track in self.tracking_history.items():
                if obj_id in current_frame_ids:
                    continue  # Already matched
                    
                # Get the last known position
                last_pos = track['positions'][-1]
                curr_pos = obj['centroid']
                
                # Calculate Euclidean distance
                dist = np.sqrt((last_pos[0] - curr_pos[0])**2 + (last_pos[1] - curr_pos[1])**2)
                
                if dist < min_dist and dist < max_distance:
                    min_dist = dist
                    best_match = obj_id
            
            if best_match is not None:
                # Update existing track
                self.tracking_history[best_match]['positions'].append(obj['centroid'])
                self.tracking_history[best_match]['bbox'] = obj['bbox']
                self.tracking_history[best_match]['frames_tracked'] += 1
                self.tracking_history[best_match]['last_seen'] = 0
                obj['id'] = best_match
                current_frame_ids.add(best_match)
            else:
                # Create new track
                self.tracking_history[self.next_object_id] = {
                    'positions': [obj['centroid']],
                    'bbox': obj['bbox'],
                    'frames_tracked': 1,
                    'last_seen': 0
                }
                obj['id'] = self.next_object_id
                current_frame_ids.add(self.next_object_id)
                self.next_object_id += 1
        
        # Update tracks that weren't matched in this frame
        tracks_to_remove = []
        for obj_id in self.tracking_history:
            if obj_id not in current_frame_ids:
                self.tracking_history[obj_id]['last_seen'] += 1
                # Remove tracks that haven't been seen for a while
                if self.tracking_history[obj_id]['last_seen'] > 30:  # ~1 second at 30fps
                    tracks_to_remove.append(obj_id)
        
        # Clean up old tracks
        for obj_id in tracks_to_remove:
            del self.tracking_history[obj_id]
            
        return objects
    
    def create_visualization(self, frame, objects, fg_mask=None, hog_image=None):
        """Create a visualization of the detected objects"""
        output = frame.copy()
        
        # Draw bounding boxes and track IDs
        for obj in objects:
            x, y, w, h = obj['bbox']
            obj_id = obj.get('id', -1)
            
            # Draw bounding box
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw ID
            cv2.putText(output, f"ID: {obj_id}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw trajectory if we have history
            if obj_id in self.tracking_history and len(self.tracking_history[obj_id]['positions']) > 1:
                positions = self.tracking_history[obj_id]['positions']
                for i in range(1, len(positions)):
                    # Draw line connecting consecutive positions
                    pt1 = positions[i - 1]
                    pt2 = positions[i]
                    cv2.line(output, pt1, pt2, (255, 0, 0), 2)
        
        return output

    def process_frame(self, frame):
        """Process a single frame with all detection steps"""
        # Step 1: Background Subtraction
        fg_mask, fg_mask_binary = self.apply_background_subtraction(frame)
        
        # Step 2: Object Detection
        objects = self.detect_objects(frame, fg_mask_binary)
        
        # Step 3: Object Tracking
        tracked_objects = self.track_objects(objects, frame)
        
        # Step 4: Feature Extraction
        hog_image = None
        if tracked_objects:
            largest_obj = max(tracked_objects, key=lambda obj: obj['area'])
            x, y, w, h = largest_obj['bbox']
            roi = frame[y:y+h, x:x+w]
            if roi.size > 0:
                _, hog_image = self.extract_hog_features(roi)
        
        # Create visualization
        output_frame = self.create_visualization(frame, tracked_objects, fg_mask, hog_image)
        
        return {
            'original': frame,
            'fg_mask': fg_mask,
            'objects': tracked_objects,
            'hog_image': hog_image,
            'output': output_frame
        }

def load_video(video_path):
    """
    Loads a video from a path.
    Returns a VideoCapture object.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Error: Could not open video at {video_path}")
    
    # Display basic video information
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video loaded successfully: {video_path}")
    print(f"Dimensions: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Frame count: {frame_count}")
    
    # Display the first frame in a window
    ret, first_frame = cap.read()
    if ret:
        cv2.imshow("First Frame", first_frame)
        cv2.waitKey(1000)  # Display for 1 second
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame
    
    return cap

def process_video(cap, max_frames=100, display_interval=10, show_intermediate=True, progress_bar=True):
    """
    Process the video frame by frame
    
    Args:
        cap: OpenCV VideoCapture object
        max_frames: Maximum number of frames to process
        display_interval: Show progress every N frames
        show_intermediate: Show intermediate results in windows
    
    Returns:
        processed_frames: List of processed frames
    """
    # Initialize surveillance system
    surveillance = VideoSurveillanceSystem(
        history=500,
        var_threshold=50,
        detect_shadows=True,
        min_area=300
    )
    
    frame_count = 0
    processed_frames = []
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = min(max_frames, total_frames)
    
    print(f"Processing video: {max_frames} frames at {fps:.2f} FPS")
    start_time = time.time()
    
    # Create windows for intermediate results if needed
    if show_intermediate:
        cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Foreground Mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Tracked Objects", cv2.WINDOW_NORMAL)
        cv2.namedWindow("HOG Features", cv2.WINDOW_NORMAL)
        
    # Create a progress bar for terminal
    if progress_bar:
        # Print an empty progress bar
        bar_width = 50
        sys.stdout.write("[%s] %d%% (0/%d frames) ETA: unknown\r" % 
                       (" " * bar_width, 0, max_frames))
        sys.stdout.flush()
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process the frame
        result = surveillance.process_frame(frame)
        processed_frames.append(result['output'])
        
        # Display progress
        if frame_count % display_interval == 0:
            print(f"Processing frame {frame_count}/{max_frames}")
            
            # Display intermediate results in windows
            if show_intermediate:
                cv2.imshow("Original", frame)
                cv2.imshow("Foreground Mask", result['fg_mask'])
                cv2.imshow("Tracked Objects", result['output'])
                
                # If we have HOG features, display them
                if result['hog_image'] is not None:
                    cv2.imshow("HOG Features", result['hog_image'])
                
                # Give time for the windows to update and check for key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Show stats about objects
            print(f"Detected {len(result['objects'])} objects")
            for i, obj in enumerate(result['objects']):
                print(f"  Object {i+1}: ID={obj.get('id', -1)}, Area={obj['area']}")
        
        # Update progress bar
        if progress_bar and max_frames > 10:
            elapsed_time = time.time() - start_time
            progress = frame_count / max_frames
            bar_width = 50
            bar_filled = int(bar_width * progress)
            
            # Calculate ETA
            if frame_count > 0:
                eta = (elapsed_time / frame_count) * (max_frames - frame_count)
                eta_min = int(eta / 60)
                eta_sec = int(eta % 60)
                
                sys.stdout.write("[%s%s] %d%% (%d/%d frames) ETA: %dm %ds\r" % 
                               ("=" * bar_filled, " " * (bar_width - bar_filled), 
                                int(progress * 100), frame_count, max_frames,
                                eta_min, eta_sec))
            else:
                sys.stdout.write("[%s%s] %d%% (%d/%d frames) ETA: calculating...\r" % 
                               ("=" * bar_filled, " " * (bar_width - bar_filled), 
                                int(progress * 100), frame_count, max_frames))
            
            sys.stdout.flush()
        
        frame_count += 1
    
    cap.release()
    
    # Close intermediate windows
    if show_intermediate:
        cv2.destroyWindow("Original")
        cv2.destroyWindow("Foreground Mask")
        cv2.destroyWindow("HOG Features")
        # Keep the tracked objects window open
    
    # Complete the progress bar
    if progress_bar and max_frames > 10:
        sys.stdout.write("\n")
    
    # Report processing time
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    fps_processing = frame_count / total_time if total_time > 0 else 0
    
    print(f"Processed {frame_count} frames in {minutes}m {seconds}s ({fps_processing:.2f} FPS)")
    return processed_frames, surveillance

def interactive_video_player(processed_frames, window_name="Surveillance Output", fps=30):
    """Interactive video player with controls"""
    if not processed_frames:
        print("No frames to display")
        return
        
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    frame_idx = 0
    paused = False
    total_frames = len(processed_frames)
    
    print("\nInteractive Video Player Controls:")
    print("  Space: Pause/Play")
    print("  A/D: Previous/Next Frame")
    print("  Q or ESC: Quit")
    
    while True:
        if frame_idx >= total_frames:
            frame_idx = 0  # Loop back to start
            
        frame = processed_frames[frame_idx]
        
        # Add control instructions to the frame
        info_frame = frame.copy()
        cv2.putText(info_frame, "Space: Pause/Play, A/D: Prev/Next, Q: Quit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(info_frame, f"Frame: {frame_idx}/{total_frames-1}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow(window_name, info_frame)
        
        # Handle key presses
        key = cv2.waitKey(0 if paused else int(1000/fps))
        
        if key == ord(' '):  # Space bar
            paused = not paused
        elif key == 83 or key == ord('d'):  # Right arrow or 'd'
            frame_idx = min(frame_idx + 1, total_frames - 1)
        elif key == 81 or key == ord('a'):  # Left arrow or 'a'
            frame_idx = max(frame_idx - 1, 0)
        elif key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif not paused:
            frame_idx += 1
    
    cv2.destroyAllWindows()

def save_video(processed_frames, output_path="surveillance_output.mp4", fps=30):
    """Save processed frames to a video file"""
    if not processed_frames:
        print("No frames to save")
        return
        
    height, width = processed_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in processed_frames:
        out.write(frame)
    
    out.release()
    print(f"Video saved to {output_path}")

def analyze_motion_patterns(processed_frames, surveillance_system):
    """Analyze motion patterns from tracked objects and display results in windows"""
    all_tracks = []
    for obj_id, track in surveillance_system.tracking_history.items():
        if len(track['positions']) > 5:  # Only consider objects tracked for at least 5 frames
            all_tracks.append({
                'id': obj_id,
                'positions': track['positions'],
                'frames_tracked': track['frames_tracked']
            })
    
    if not all_tracks:
        print("No tracks available for analysis")
        return
        
    # Create a plot of trajectories and save it
    plt.figure(figsize=(12, 8))
    for track in all_tracks:
        x_points = [pos[0] for pos in track['positions']]
        y_points = [pos[1] for pos in track['positions']]
        plt.plot(x_points, y_points, '-o', linewidth=2, markersize=4, 
                 label=f"Object {track['id']}")
    
    plt.title("Motion Trajectories")
    plt.xlabel("X position (pixels)")
    plt.ylabel("Y position (pixels)")
    plt.gca().invert_yaxis()  # Invert Y-axis to match image coordinates
    plt.legend()
    plt.grid(True)
    plt.savefig("motion_trajectories.png")
    print("Motion trajectories saved to motion_trajectories.png")
    
    # Generate a heat map of activity
    if processed_frames:
        height, width = processed_frames[0].shape[:2]
        heatmap = np.zeros((height, width))
        
        for track in all_tracks:
            for pos in track['positions']:
                x, y = int(pos[0]), int(pos[1])
                if 0 <= x < width and 0 <= y < height:
                    # Create a gaussian "heat" around each point
                    for dy in range(-15, 16):
                        for dx in range(-15, 16):
                            if dx*dx + dy*dy <= 15*15:  # Circle with radius 15
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < width and 0 <= ny < height:
                                    heatmap[ny, nx] += 1
        
        # Normalize and create heat map
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = heatmap.astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Blend with the first frame for context
        blended = cv2.addWeighted(processed_frames[0], 0.7, heatmap, 0.3, 0)
        
        # Display the heatmap
        cv2.imshow("Activity Heat Map", blended)
        cv2.waitKey(0)  # Wait for a key press
        
        # Save the heatmap
        cv2.imwrite("activity_heatmap.png", blended)
        print("Activity heat map saved to activity_heatmap.png")

def main():
    # Add ability to pass command line arguments
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        print(f"Using video path from command line: {video_path}")
    else:
        # Ask user for video path
        default_path = "example_vid_1.mp4"
        print("Optional paths: example_vid_1.mp4, example_vid_2.mp4, example_vid_3.mp4")
        video_path = input(f"Enter video path (press Enter for default: {default_path}): ") or default_path
    
    try:
        # 1. Load the video
        print(f"Loading video from: {video_path}")
        cap = load_video(video_path)
        
        # 2. Process the video
        frame_input = input("Enter maximum frames to process (default: 100, enter 'all' for full video): ")
        if frame_input.lower() == 'all':
            max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Processing all {max_frames} frames...")
        else:
            max_frames = int(frame_input or "100")
            
        # Determine appropriate display interval based on number of frames
        display_interval = max(1, max_frames // 10)  # Show approximately 10 updates
        
        print("\nProcessing video...")
        processed_frames, surveillance = process_video(cap, max_frames=max_frames, display_interval=display_interval)
        
        # 3. Save the video
        save_video(processed_frames)
        
        # 4. Display the video in interactive player
        print("\nStarting interactive video player...")
        interactive_video_player(processed_frames)
        
        # 5. Perform motion analysis if requested
        if input("\nPerform motion analysis? (y/n): ").lower().startswith('y'):
            analyze_motion_patterns(processed_frames, surveillance)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    main()