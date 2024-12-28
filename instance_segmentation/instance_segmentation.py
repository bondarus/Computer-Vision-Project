import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Load the YOLO model
model = YOLO("yolo11n-seg.pt")  # segmentation model

# Load the video
cap = cv2.VideoCapture("./videos/video1.mp4")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
out = cv2.VideoWriter("./output/instance-segmentation.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

# Define the classes to detect 
target_classes = ["person", "bicycle", "car"]

while True:
    ret, im0 = cap.read()
    if not ret:
        break

    # Perform instance segmentation
    results = model.predict(im0)
    annotator = Annotator(im0, line_width=2)

    if results[0].masks is not None:
        clss = results[0].boxes.cls.cpu().tolist()
        masks = results[0].masks.xy
        for mask, cls in zip(masks, clss):
            class_name = model.model.names[int(cls)]
            if class_name in target_classes:
                color = colors(int(cls), True)
                txt_color = annotator.get_txt_color(color)
                annotator.seg_bbox(mask=mask, mask_color=color, label=class_name, txt_color=txt_color)

    out.write(im0)
    cv2.imshow("instance-segmentation", im0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
