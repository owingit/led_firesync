import os
import cv2
import tqdm


def blob_detect(f, trial, visualize_detections, dir):
    reconstruction_data = {'frame time': [], 'object time': [], 'object x position': [], 'object y position': []}
    cap = cv2.VideoCapture(f)

    if not cap.isOpened():
        print(f"Failed to open video: {f}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    reconstruction_data['fps'] = fps
    nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=nFrames, desc=f"Processing Trial {trial}", unit="frame")
    framec = 0
    while cap.isOpened():
        ret, frame = cap.read()
        framec+=1
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # === SimpleBlobDetector parameters ===
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = False
        params.blobColor = 255

        params.minThreshold = 30
        params.maxThreshold = 255

        params.filterByArea = True
        params.minArea = 12     # small blobs (LED/firefly)
        params.maxArea = 200   # ignore large blobs

        params.filterByCircularity = False
        params.filterByInertia = False
        params.filterByConvexity = False

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)

        frame_time = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
        reconstruction_data['frame time'].append(frame_time)

        if visualize_detections:
            frame_with_boxes = frame.copy()
        for k in keypoints:
            x, y = k.pt
            reconstruction_data['object x position'].append(x)
            reconstruction_data['object y position'].append(y)
            reconstruction_data['object time'].append(frame_time)  
            if visualize_detections:
                size = k.size
                top_left = (int(x - size / 2), int(y - size / 2))
                bottom_right = (int(x + size / 2), int(y + size / 2))
                cv2.rectangle(frame_with_boxes, top_left, bottom_right, (0, 0, 255), 2)

        # Save annotated frame
        if visualize_detections:
            if len(keypoints) > 0:
                op = os.path.join(dir, f'trial_{trial}_xy_frame_{framec}.png')
                cv2.imwrite(op, frame_with_boxes)
        pbar.update(1)

    pbar.close()
    cap.release()
    return reconstruction_data
