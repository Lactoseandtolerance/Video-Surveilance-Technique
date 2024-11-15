import cv2
import numpy as np
from skimage.feature import hog
from sklearn.cluster import KMeans

# Load video
video_path = '/Users/professornirvar/Downloads/Night Video with Ambient Lighting Sample Footage -Wasp Pro 4 0 mp4 - SCW - Security Technology (720p, h264, youtube) (1).mp4'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Initialize background subtractor
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Define utility functions
def apply_hog_descriptor(frame):
    """Extract HOG (Histogram of Oriented Gradients) features."""
    # Ensure the input is grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    # Extract HOG features and the visualization
    features, hog_image = hog(
        gray_frame,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        feature_vector=True
    )
    return features, hog_image

def selective_search(frame):
    """Apply Selective Search for region proposals."""
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(frame)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    return rects[:100]  # Return top 100 region proposals for efficiency

def draw_bounding_boxes(frame, boxes, fg_mask):
    """Draw bounding boxes on the frame, tuned for better background inclusion."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 300:  # Adjusted for sensitivity
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Step 1: Adaptive Background Modeling
    fg_mask = background_subtractor.apply(frame)

    # Apply morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Extract foreground objects
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 500:  # Ignore small regions
            continue
        x, y, w, h = cv2.boundingRect(contour)
        roi = frame[y:y+h, x:x+w]

        # Step 2: Selective Search
        proposals = selective_search(roi)

        # Step 3: Apply HOG and Relative Orientation Features
        _, hog_image = apply_hog_descriptor(roi)
        kmeans = KMeans(n_clusters=2)  # Example clustering to distinguish objects
        labels = kmeans.fit_predict(hog_image.reshape(-1, 1))

        # Step 4: Spatial Pyramid Matching (Simplified for Visualization)
        levels = 2
        step_x = w // levels
        step_y = h // levels
        for i in range(levels):
            for j in range(levels):
                start_x, start_y = x + j * step_x, y + i * step_y
                end_x, end_y = start_x + step_x, start_y + step_y
                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 1)

        # Draw region proposals on the frame using the updated function
        frame = draw_bounding_boxes(frame, proposals, fg_mask)

    # Display the frame
    cv2.imshow('Video Surveillance Pipeline', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
