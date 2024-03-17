import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import numpy as np

# ANSI escape codes for colors
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"
    CYAN = "\033[96m"

# Define focus zone of the camera. Can be dynamic in nature
ZONE_POLYGON = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
])

DETECTION_CLASS = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
}

VIDEO_SOURCE = "./MOT16-05-raw.webm"

# DETECTION_CLASS = [0, 1]

def printDetections(detected_objects):
    countDict = {}

    # Count the instances of each key
    for key in DETECTION_CLASS.keys():
        countDict[key] = 0  # Initialize the count for each key

    # Update counts based on detected objects
    for obj in detected_objects:
        if obj in countDict:
            countDict[obj] += 1

    for key, value in countDict.items():
        print(Colors.BLUE + f"Number of {DETECTION_CLASS[key]}:" + Colors.RESET + Colors.GREEN + f" {value}" + Colors.RESET)


def parseArguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Yolov8 live")
    parser.add_argument("--webcam-resolution", default=[1280, 720], nargs=2, type=int)
    args = parser.parse_args()
    return args

def main():
    # Start video capture
    args = parseArguments()
    frameWidth, frameHeight = args.webcam_resolution

    print(Colors.GREEN + "Starting webcam..." + Colors.RESET)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)

    # Set YOLO weights
    model = YOLO("yolov8l.pt")

    # Set bounding-boxes attributes
    boxAnnotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)

    zonePolygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zonePolygon, frame_resolution_wh=tuple(args.webcam_resolution))
    zoneAnnotator = sv.PolygonZoneAnnotator(
        zone=zone, 
        color=sv.Color.red(),
        thickness=2,
        text_thickness=2,
        text_scale=2
        )
    
    try:
        while True:

            ret, frame = cap.read()

            result = model(
                        frame,      # To test the model, put the path of the video instead of camera frame
                        agnostic_nms=True,      #Turning on agnostic_nms to avoid double detection
                        classes=list(DETECTION_CLASS.keys()), 
                        conf = 0.75
                    )[0]            # Remove the [0] while testing the video
            
            detections = sv.Detections.from_yolov8(result)
            printDetections(detections.class_id)

            labels = [
                f"{model.model.names[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, _
                in detections
            ]

            frame = boxAnnotator.annotate(
                scene=frame, 
                detections=detections,
                labels=labels
            )

            zone.trigger(detections=detections)
            frame = zoneAnnotator.annotate(scene=frame)

            cv2.imshow("yolov8", frame)

            if(cv2.waitKey(30) == 27):
                break

    except KeyboardInterrupt:
        print(Colors.RED + "Turning of the Webcam..." + Colors.RESET)

if __name__ == "__main__":
    main()