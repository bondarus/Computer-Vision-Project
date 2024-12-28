import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import numpy as np

# Load the YOLOv8 segmentation model
model = YOLO('yolov8n-seg.pt')  # Ensure this is a segmentation model

# Define the class IDs for vehicles and cyclists based on the model's training data
target_classes = [1, 2, 3, 5, 7]  # Adjust based on your model's class indices

# Open the input video
input_video_path = './videos/video1.mp4'
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print(f"Error: Cannot open video file {input_video_path}")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
output_video_path = './output/labeled_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    # Perform instance segmentation
    results = model(frame)

    # Initialize annotator for drawing
    annotator = Annotator(frame)

    # Iterate over detected instances
    for result in results:
        if result.boxes is not None:
            for box, mask in zip(result.boxes, result.masks.data):
                cls = int(box.cls)  # Class ID
                if cls in target_classes:
                    # Convert mask to binary and resize to frame dimensions
                    mask = mask.cpu().numpy().astype(np.uint8) * 255
                    mask_resized = cv2.resize(mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

                    # Create colored mask
                    mask_color = colors(cls)
                    colored_mask = np.zeros_like(frame, dtype=np.uint8)
                    colored_mask[mask_resized > 0] = mask_color

                    # Overlay the mask on the frame
                    frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)

                    # Extract bounding box coordinates and draw the label
                    x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())  # Convert coordinates to integers
                    label = f"{model.names[cls]}"
                    annotator.box_label((x1, y1, x2, y2), label)

    # Get the annotated frame
    labeled_frame = annotator.result()

    # Write the frame to the output video
    out.write(labeled_frame)

    # Display the frame (optional)
    cv2.imshow('Instance Segmentation', labeled_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved to: {output_video_path}")
