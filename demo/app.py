from flask import Flask, render_template, send_from_directory, request, redirect, url_for
import os
import video_processing
import models
from time import time
from tqdm import tqdm

app = Flask(__name__, static_folder="processed")

# Define the directory where your videos are stored
VIDEO_FOLDER = 'C:/Users/chuag/OneDrive - Nanyang Technological University/Desktop/BCG 4.2/FYP/code/demo/videos'
MODEL_FOLDER = 'C:/Users/chuag/OneDrive - Nanyang Technological University/Desktop/BCG 4.2/FYP/code/demo/models'
OUTPUT_FOLDER = 'C:/Users/chuag/OneDrive - Nanyang Technological University/Desktop/BCG 4.2/FYP/code/demo/processed'

video_name = None

@app.route('/')
def index():
    # Get a list of all video files in the 'videos' folder
    videos = os.listdir(os.path.join(os.path.dirname(__file__), VIDEO_FOLDER))
    return render_template('index.html', videos=videos)

@app.route('/video', methods=['GET'])
def play_video():
    global video_name
    models = os.listdir(os.path.join(os.path.dirname(__file__), MODEL_FOLDER))
    video_name = request.args.get('video')
    return render_template('play.html', video_name=video_name, models=models)

@app.route('/video/<path:filename>')
def serve_video(filename):
    return send_from_directory(VIDEO_FOLDER, filename)

# Route for the loading page
@app.route('/results', methods = ["GET"])
def loading():
    model_name = request.args.get('model')
    model_path = os.path.join(os.path.dirname(__file__), MODEL_FOLDER, model_name)

    video_path = os.path.join(os.path.dirname(__file__), VIDEO_FOLDER, video_name)

    out_path = os.path.join(os.path.dirname(__file__), OUTPUT_FOLDER, "out.mp4")

    # prepare model
    model = models.prepare_model(model_path)

    # extracting frames
    frames = video_processing.extract_frames(video_path)

    tailored_frames = []
    time_taken = []
    predictions = []

    # extracting face crops
    for frame in tqdm(frames):
        cropped_face = video_processing.crop_faces(frame)
        
        if cropped_face:
            face_crop, face_loc = cropped_face
            face_tensor = video_processing.crop_to_tensor(face_crop)

            # run model predictions and time taken
            t1 = time()
            pred = models.predict(model, face_tensor)
            predictions.append(pred)
            t2 = time()
            t_delta = t2 - t1
            time_taken.append(t_delta)

            frame = video_processing.apply_prediction(frame, pred, face_loc)
        
        tailored_frames.append(frame)
    
    video_processing.frames_to_video(tailored_frames, output_video_path=out_path, fps = 30)

    total_time = round(sum(time_taken), 4)
    avg_time = round(total_time / len(time_taken), 4)
    
    if sum(predictions) < len(tailored_frames)/2:
        classification = "Real"
    else:
        classification = "Fake"

    return render_template('results.html', total_time = total_time, avg_time = avg_time, video_name = "out.mp4", classification = classification)

@app.route('/results/<path:filename>')
def processed_video(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, conditional = True)

if __name__ == '__main__':
    app.run(debug=True)
