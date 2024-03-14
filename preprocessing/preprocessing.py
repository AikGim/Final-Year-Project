import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import cv2
import random
from tqdm import tqdm
from mtcnn import MTCNN

'''
In this preprocessing script, we will extract frames from a folder of videos and extract them into a single folder.
Faces will be extracted into a separate folder.
'''

CONTINUE_FROM = 0

EXT_FRAMES = True
EXT_FACES = True

LABEL = "fake"
TARGET = "NeuralTextures"
COMPRESSION = "c23"

CROP_METHOD = "HAAR" # "MTCNN" otherwise it's "HAAR"

HEADER = "manipulated_sequences"

SOURCE_PATH = f"C:/Users/chuag/OneDrive - Nanyang Technological University/Desktop/BCG 4.2/FYP/code/data/FF++/{HEADER}/{TARGET}/{COMPRESSION}/videos"
OUTPUT_PATH = f"C:/Users/chuag/OneDrive - Nanyang Technological University/Desktop/BCG 4.2/FYP/code/data/FF++/processed/{COMPRESSION}/{LABEL}/{TARGET}"

def extract_frames(filename, vidpath, outpath, interval=30, resize = None):
    '''
    input: a video found using file path
    output: frames of video as images in specified folder based on interval specified
    '''

    cap = cv2.VideoCapture(vidpath)
    frame_count = 0

    while cap.isOpened():
        ret = cap.grab()

        if not ret:
            break

        if frame_count % interval == 0:
            ret, frame = cap.retrieve()

            frame_path = f"{outpath}/{filename}_{frame_count:05d}.jpg"
            if resize:
                frame = cv2.resize(frame, resize)
            cv2.imwrite(frame_path, frame)

        frame_count += 1

    cap.release()


def sample_frames(filename, vidpath, outpath, sample=10, resize=None):
    '''
    input: a video found using file path
    output: randomly extract frames of input video as images in specified folder based on sample size provided
    '''
    # Open video file
    cap = cv2.VideoCapture(vidpath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Generate random frame indices
    if sample >= total_frames:
        sample = total_frames

    random_indices = random.sample(range(total_frames), sample)
    random_indices.sort()

    # Extract random frames
    frame_count = 0
    index = 0

    while cap.isOpened() and index < sample:
        ret = cap.grab()

        if not ret:
            break

        if frame_count == random_indices[index]:
            ret, frame = cap.retrieve()

            # Save frame
            frame_path = os.path.join(outpath, f"{filename}_{index:05d}.jpg")

            if resize:
                frame = cv2.resize(frame, resize)
            
            cv2.imwrite(frame_path, frame)

            index += 1

        frame_count += 1

    cap.release()

def crop_faces_mtcnn(detector, filename, imgpath, outpath, xy_expansion = 0.2, wh_expansion = 1.4):

    img = cv2.imread(imgpath)

    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(rgb_image)

    # (x, y) references the top-left coordinate of the bounding box found by face detector
    # w and h references height and width of the bounding box
    for i, face in enumerate(faces):
        x, y, w, h = face['box']

        # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Calculate the expanded bounding box dimensions
        expanded_w = int(w * wh_expansion)
        expanded_h = int(h * wh_expansion)

        size = max(expanded_w, expanded_h)

        # Calculate the new coordinates for the expanded bounding box
        x2 = max(0, x + w // 2 - size // 2)
        y2 = max(0, y + h // 2 - size // 2)

        # Crop the expanded area
        face_crop = img[y2:y2+size, x2:x2+size]

        writepath = f"{outpath}/{filename}_{i}.jpg"
        cv2.imwrite(writepath, face_crop)

        return # just take the first index as face


def crop_faces_haar(face_cascade, filename, imgpath, outpath, xy_expansion = 0.2, wh_expansion = 1.4, scaleFactor = 1.3, min_neighbours = 10):

    img = cv2.imread(imgpath)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
    # Detect faces 
    faces = face_cascade.detectMultiScale(gray, scaleFactor = scaleFactor, minNeighbors=min_neighbours)

    # (x, y) references the top-left coordinate of the bounding box found by face cascade
    # w and h references height and width of the bounding box

    # Find the face with the largest bounding box
    # largest_face = None
    # largest_area = 0
    # for (x, y, w, h) in faces:
    #     area = w * h
    #     print(area)
    #     if area > largest_area:
    #         largest_area = area
    #         largest_face = (x, y, w, h)

    # if largest_face is not None:
    #     x, y, w, h = largest_face

    # Sort faces by their area
    sorted_faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True) # x[2] = w; x[3] = h

    for i, (x, y, w, h) in enumerate(sorted_faces):

        # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # Define the new dimensions for the bounding box
        expanded_x = max(0, x - int(w * xy_expansion))  # Expand x-coordinate to the left
        expanded_y = max(0, y - int(h * xy_expansion))  # Expand y-coordinate upwards
        expanded_w = min(img.shape[1] - expanded_x, int(w * wh_expansion))  # Expand width
        expanded_h = min(img.shape[0] - expanded_y, int(h * wh_expansion))  # Expand height

        # Crop the expanded area
        face_crop = img[expanded_y:expanded_y + expanded_h, expanded_x:expanded_x + expanded_w]

        # cv2.imshow("face",faces) 
        writepath = f"{outpath}/{filename}_{i}.jpg"
        cv2.imwrite(writepath, face_crop)

        return # just take the first index as face

##### MODEL SETTINGS #####

if CROP_METHOD == "MTCNN":
    crop_faces = crop_faces_mtcnn
    model = MTCNN()

else:
    crop_faces = crop_faces_haar
    model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if __name__ == '__main__':

    print("< < < Initiating Frames + Face Extraction . . . > > >")
    print("Source folder: %s" % os.path.basename(SOURCE_PATH))
    print("Destination folder: %s" % os.path.basename(OUTPUT_PATH))

    # Create output folder for frames
    frames_folder = OUTPUT_PATH + "/frames"
    os.makedirs(frames_folder, exist_ok=True)

    if EXT_FRAMES:
        print("< < < Commencing Frame Extraction  > > >")
        # Iterate through video files in the input folder
        for filename in tqdm(os.listdir(SOURCE_PATH)[CONTINUE_FROM:]):
            if filename.endswith('.mp4'):
                vidpath = os.path.join(SOURCE_PATH, filename)
                vidname = os.path.splitext(filename)[0]

            # extract_frames(vidname, vidpath, frames_folder, interval = 30)

            sample_frames(vidname, vidpath, frames_folder, sample = 10)
    
    if EXT_FACES:
        print("< < < Commencing Face Extraction  > > >")

        # Create output folder for face crops
        faces_folder = OUTPUT_PATH + "/faces"
        os.makedirs(faces_folder, exist_ok=True)

        # Iterate through image files in the newly created frames folder
        for filename in tqdm(os.listdir(frames_folder)):
            if filename.endswith('.jpg'):
                imgpath = os.path.join(frames_folder, filename)
                imgname = os.path.splitext(filename)[0]

            crop_faces(model, imgname, imgpath, faces_folder)
        

# video_path = "test_folder/011_00300.jpg"
# sample_frames("frame", video_path, "test_folder")

# img = crop_faces(model, "face", "test_folder/011_00300.jpg", "test_folder")

# cv2.imshow('img', img)
# cv2.waitKey()

# # makes folder if doesn't exist
# if not os.path.exists(outpath):
#     os.makedirs(outpath)