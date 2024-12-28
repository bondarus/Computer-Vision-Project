import cv2
import imutils
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

def license_plate_segmentation(im0):
    roi = im0[y2:y1, x2:x1]
    if roi.size != 0:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 13, 15, 15)
        edged = cv2.Canny(gray, 30, 200)

        contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        # Initialize the license plate contour
        license_plate_contour = None
        for contour in contours:
            # Approximate the contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * peri, True)

            # Check if the approximated contour has four points
            if len(approx) == 4:
                license_plate_contour = approx
                break

        # Draw the contour on the original image if a valid contour is found
        if license_plate_contour is not None:
            license_plate_contour[:, 0, 0] += x2
            license_plate_contour[:, 0, 1] += y2
            cv2.drawContours(im0, [license_plate_contour], -1, (0, 255, 0), 3)

    return im0


# Load the YOLO model
model = YOLO("license_plate_detector.pt")

# Load the video
cap = cv2.VideoCapture("./videos/video1.mp4")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
out = cv2.VideoWriter("./output/license_plate_detection.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

while True:
    ret, im0 = cap.read()
    if not ret:
        break

    #im0 = cv2.GaussianBlur(im0, (5, 5), 1)

    # Perform license plate detection
    results = model.predict(im0)
    annotator = Annotator(im0, line_width=2)

    if results[0].boxes is not None:
        clss = results[0].boxes.cls.cpu().tolist()
        boxes = results[0].boxes.xyxy
        for box, cls in zip(boxes, clss):
            class_name = model.model.names[int(cls)]
            if class_name == "license_plate":

                # Tighten the bounding box if necessary (adjust margins if required)
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Optional: Expand slightly inward to tighten
                x_margin = 0.06  # Adjust margin percentage
                y_margin = 0.20  # Adjust margin percentage
                width = x2 - x1
                height = y2 - y1

                x1 = max(0, x1 + int(x_margin * width))
                y1 = max(0, y1 + int(y_margin * height))
                x2 = min(im0.shape[1], x2 - int(x_margin * width))
                y2 = min(im0.shape[0], y2 - int(y_margin * height))

                color = colors(int(cls), True)
                txt_color = annotator.get_txt_color(color)

                annotator.box_label(box=[x1, y1, x2, y2], label=class_name, color=color, txt_color=txt_color)

                im0 = license_plate_segmentation(im0)

    out.write(im0)
    cv2.imshow("license_plate_detection", im0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
