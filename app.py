from flask import Flask, render_template, request, redirect, session, jsonify, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
import os
import cv2
from db import init_mysql, get_user_by_username, insert_user
from predict import run_deepfake_detection  # Import predict functionality

app = Flask(__name__)
app.secret_key = "temp123"  # For development purposes only

# Initialize MySQL
init_mysql(app)

# Folders
UPLOAD_FOLDER = "uploads"
FRAMES_FOLDER = "frames"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def home():
    if 'loggedin' in session:
        return redirect('/upload')
    return redirect('/login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password_input = request.form['password']
        user = get_user_by_username(username)

        if user and check_password_hash(user['password'], password_input):
            session['loggedin'] = True
            session['id'] = user['id']
            session['username'] = user['username']
            return redirect('/upload')
        else:
            return render_template('login.html', message='Invalid credentials')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])

        if get_user_by_username(username):
            return render_template('register.html', message="Username already exists")
        
        insert_user(username, password)
        return redirect('/login')
    return render_template('register.html')

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if 'loggedin' not in session:
        return redirect('/login')
    
    if request.method == "POST":
        if "video" not in request.files:
            return jsonify({"message": "No video uploaded"}), 400

        video = request.files["video"]
        frame_rate = int(request.form.get("frame_rate", 5))
        video_name = os.path.splitext(video.filename)[0]
        video_path = os.path.join(UPLOAD_FOLDER, video.filename)
        video.save(video_path)

        save_folder = os.path.join(FRAMES_FOLDER, video_name)

        if os.path.exists(save_folder):
            for filename in os.listdir(save_folder):
                file_path = os.path.join(save_folder, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            os.rmdir(save_folder)

        os.makedirs(save_folder, exist_ok=True)
        extract_frames(video_path, frame_rate, save_folder)

        # Run deepfake prediction
        summary = run_deepfake_detection(save_folder)

        return jsonify({
            "message": "Video processed successfully!",
            "video_name": video_name,
            "summary": summary
        })
    
    return render_template("upload.html", username=session['username'])

@app.route("/view_frames/<video_name>")
def view_frames(video_name):
    frame_folder = os.path.join(FRAMES_FOLDER, video_name)
    if not os.path.exists(frame_folder):
        return jsonify([])

    frame_files = sorted(os.listdir(frame_folder))
    frame_urls = [f"/frames/{video_name}/{filename}" for filename in frame_files if filename.endswith(".jpg")]
    return jsonify(frame_urls)

@app.route('/frames/<video_name>/<filename>')
def serve_frame(video_name, filename):
    return send_from_directory(os.path.join(FRAMES_FOLDER, video_name), filename)

def extract_frames(video_path, frame_rate, save_folder):
    cap = cv2.VideoCapture(video_path)
    count = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    interval = max(1, fps // frame_rate)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        if count % interval == 0:
            frame_filename = os.path.join(save_folder, f"frame_{count}.jpg")
            cv2.imwrite(frame_filename, frame)
        count += 1

    cap.release()

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

if __name__ == "__main__":
    app.run(debug=True)
