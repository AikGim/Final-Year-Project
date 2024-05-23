'''
this script achieves the following
1) it first breaks down the source video into frames
2) identifies faces for each frame using bounding boxes with the help of a face detector algorithm
3) applies model prediction to each identified face for each frame
4) re-stitches the video back together with the above changes as output
'''

import os
import cv2
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

SIZE = 224 # model size

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((SIZE, SIZE), antialias = True)
    ])

def extract_frames(vidpath):
    '''
    input: a video found using file path
    output: frames of input video
    '''

    cap = cv2.VideoCapture(vidpath)

    frames = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()

    return frames

def crop_faces(frame, xy_expansion = 0.2, wh_expansion = 1.4, scaleFactor = 1.3, min_neighbours = 10):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
    # Detect faces 
    faces = face_cascade.detectMultiScale(gray, scaleFactor = scaleFactor, minNeighbors=min_neighbours)

    # (x, y) references the top-left coordinate of the bounding box found by face cascade
    # w and h references height and width of the bounding box

    # Sort faces by their area
    sorted_faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True) # x[2] = w; x[3] = h

    for x, y, w, h in sorted_faces:

        # Define the new dimensions for the bounding box
        expanded_x = max(0, x - int(w * xy_expansion))  # Expand x-coordinate to the left
        expanded_y = max(0, y - int(h * xy_expansion))  # Expand y-coordinate upwards
        expanded_w = min(frame.shape[1] - expanded_x, int(w * wh_expansion))  # Expand width
        expanded_h = min(frame.shape[0] - expanded_y, int(h * wh_expansion))  # Expand height

        face_loc = (expanded_x, expanded_y, expanded_w, expanded_h)

        # Crop the expanded area

        face_crop = frame[expanded_y:expanded_y + expanded_h, expanded_x:expanded_x + expanded_w]

        return face_crop, face_loc # just take the first index as face

def crop_to_tensor(face_crop):
    # frame_pil = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    # frame_pil = Image.fromarray(face_crop)

    # Apply transform
    tensor = transform(face_crop)

    tensor = tensor.unsqueeze(0)

    return tensor

def apply_faceloc(frame, face_loc):
    # applies prediction to the face crop frame
    x, y, w, h = face_loc

    # label positions relative to bounding box
    a, b = 10, 20

    cv2.rectangle(frame, (x, y),
        (x + w, y + h),
        (255, 0, 0), # color of frame BGR
        3)
    
    return frame

def apply_prediction(frame, pred, face_loc):
    # applies prediction to the face crop frame
    x, y, w, h = face_loc

    # label positions relative to bounding box
    a, b = 10, 20

    if pred:
        cv2.rectangle(frame, (x, y),
                (x + w, y + h),
                (0, 0, 255), # color of frame BGR
                3)
        
        # label
        font = cv2.FONT_HERSHEY_TRIPLEX
        cv2.putText(frame, "deepfake", (x + a, y + b), font, 0.9, (255, 255, 255), 2)

    else:
        cv2.rectangle(frame, (x, y),
                (x + w, y + h),
                (0, 255, 0), # color of frame BGR
                3)
        
        # label
        font = cv2.FONT_HERSHEY_TRIPLEX
        cv2.putText(frame, "real", (x + a, y + b), font, 0.9, (255, 255, 255), 2)
    
    return frame

# Function to stitch frames into a video
def frames_to_video(frames, output_video_path, fps):
    if not frames:
        print("No frames to write!")
        return
    
    # Get height and width of frames
    height, width, _ = frames[0].shape
    
    # Initialize VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'h264') # h264 codec is compatible with browsers
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Write frames to video
    for frame in tqdm(frames):
        out.write(frame)
    
    # Release VideoWriter
    cv2.destroyAllWindows()
    out.release()
    print(f"Video saved as {output_video_path}")

if __name__ == '__main__':
    import models

    model = models.prepare_model("demo\models\LITE0-KD25-pruned35-5iter-LR-FT.pth")

    path = "demo/videos/650.mp4"

    frames = extract_frames(path)

    tailored_frames = []

    for frame in tqdm(frames[:1]):

        cv2.imshow('img', frame)
        cv2.waitKey()

        # find face crop in the video
        cropped_face = crop_faces(frame)

        # if a face was detected in the frame
        if cropped_face:
            face_crop, face_loc = cropped_face

            cv2.imshow('img', face_crop)
            cv2.waitKey()

            frame = apply_faceloc(frame, face_loc)

            cv2.imshow('img', frame)
            cv2.waitKey()

            face_tensor = crop_to_tensor(face_crop)

            pred = models.predict(model, face_tensor)

            # frame = apply_prediction(frame, pred, face_loc)




        tailored_frames.append(frame)
    
    # frames_to_video(tailored_frames, output_video_path="demo/processed/out.mp4", fps = 30)