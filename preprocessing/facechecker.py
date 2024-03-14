# video_path = "C:/Users/chuag/OneDrive - Nanyang Technological University/Desktop/BCG 4.2/FYP/code/data/FF++/manipulated_sequences/FaceShifter/c40/videos/052_108.mp4"
# sample_frames("frame", video_path, "test_folder")

# img = crop_faces("face", "test_folder/frame_00000.jpg", "test_folder")

# cv2.imshow('img', img) 
# cv2.waitKey()

# # makes folder if doesn't exist
# if not os.path.exists(outpath):
#     os.makedirs(outpath)

import cv2
import os
import shutil
from tqdm import tqdm

LABEL = "real"
TARGET = "actors"
COMPRESSION = "c23"

TEST = False

SOURCE_PATH = f"C:/Users/chuag/OneDrive - Nanyang Technological University/Desktop/BCG 4.2/FYP/code/data/FF++/processed/{COMPRESSION}/{LABEL}/{TARGET}/faces"
OUTPUT_PATH = f"C:/Users/chuag/OneDrive - Nanyang Technological University/Desktop/BCG 4.2/FYP/code/data/FF++/processed/quarantine/{COMPRESSION}/{TARGET}"


def contains_face(imgpath, scaleFactor = 1.3, minNeighbours=10):
    img = cv2.imread(imgpath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbours)

    return len(faces) > 0


# Function to filter out non-face images from a directory
def filter_non_faces(image_dir, quarantine_dir):
    non_face_images = []

    # Iterate over the images in the directory
    for filename in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, filename)
        # Check if the image contains a face
        if not contains_face(image_path):
            non_face_images.append(filename)

            # Move the non-face image to the quarantine directory
            shutil.move(image_path, os.path.join(quarantine_dir, filename))

            # Optionally, you can also remove the non-face images here
            # os.remove(image_path)

    return non_face_images

if __name__ == '__main__':

    if not TEST:
        # create quarantine directory
        print("< < < Initiating Face Checking . . . > > >")
        print("Source folder: %s" % os.path.basename(SOURCE_PATH))
        print("Destination folder: %s" % os.path.basename(OUTPUT_PATH))

        os.makedirs(OUTPUT_PATH, exist_ok=True)

        # Filter out non-face images from the directory

        non_face_images = filter_non_faces(SOURCE_PATH, OUTPUT_PATH)

        # Print non-face images
        print("Non-face images:", non_face_images)
        print("Detected %s non-faces" % len(non_face_images)) 

    else:
        # TEST #

        SOURCE = "test_folder"
        QUARANTINE = "quarantine"

        non_face_images = filter_non_faces(SOURCE, QUARANTINE)

        # Print non-face images
        print("Non-face images:", non_face_images)
        print("Detected %s non-faces" % len(non_face_images)) 