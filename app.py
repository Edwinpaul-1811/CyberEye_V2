from flask import Flask, render_template, request, redirect, session, jsonify, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
import os
import cv2
from db import init_mysql, get_user_by_identifier, insert_user, insert_video, insert_frame, insert_face, insert_analysis_result, insert_video_result, get_videos_by_user, get_video_filename, delete_video_by_id
from predict import run_deepfake_detection

app = Flask(__name__)
app.secret_key = "temp123"

init_mysql(app)

UPLOAD_FOLDER = "uploads"
FRAMES_FOLDER = "frames"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def home():
    if 'loggedin' in session:
        return redirect('/index')
    return redirect('/login')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        identifier = request.form['identifier']
        password_input = request.form['password']
        user = get_user_by_identifier(identifier)

        if user and check_password_hash(user['password_hash'], password_input):
            session['loggedin'] = True
            session['id'] = user['user_id']
            session['username'] = user['username']
            return redirect('/index')
        else:
            return render_template('login.html', message='Invalid credentials')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            return render_template('register.html', message="Passwords do not match")
        
        if get_user_by_identifier(username) or get_user_by_identifier(email):
            return render_template('register.html', message="Username or email already exists")
        
        password_hash = generate_password_hash(password)
        insert_user(username, email, password_hash)
        return redirect('/login')
    return render_template('register.html')

@app.route("/index", methods=["GET", "POST"])
def index():
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

        cap = cv2.VideoCapture(video_path)
        duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        video_id = insert_video(session['id'], video.filename, video_path, frame_rate, duration, os.path.splitext(video.filename)[1][1:])

        save_folder = os.path.join(FRAMES_FOLDER, video_name)
        if os.path.exists(save_folder):
            for filename in os.listdir(save_folder):
                os.remove(os.path.join(save_folder, filename))
            os.rmdir(save_folder)
        os.makedirs(save_folder, exist_ok=True)

        frame_ids = extract_frames(video_path, frame_rate, save_folder, video_id)

        # Run deepfake prediction and insert faces + analysis
        summary, total_faces, frame_statuses = run_deepfake_detection(save_folder, frame_ids)


        # Insert overall video result
        insert_video_result(video_id, summary['deepfake_percentage'], len(frame_ids), total_faces)

        return jsonify({
            "message": "Video processed successfully!",
            "video_name": video_name,
            "summary": summary,
            "frame_statuses": frame_statuses  # <-- NEW
        })

    
    return render_template("index.html", username=session['username'])

@app.route('/uploads/videos/<filename>')
def serve_uploaded_video(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/uploads')
def uploads():
    if 'loggedin' not in session:
        return redirect('/login')

    videos = get_videos_by_user(session['id'])

    # Only keep videos whose files exist
    existing_videos = [
        v for v in videos if os.path.exists(os.path.join(UPLOAD_FOLDER, v['filename']))
    ]

    return render_template('uploads.html', videos=existing_videos, username=session['username'])

@app.route('/delete_video/<int:video_id>', methods=['POST'])
def delete_video(video_id):
    if 'loggedin' not in session:
        return redirect('/login')

    video = get_video_filename(video_id, session['id'])
    delete_video_by_id(video_id, session['id'])
    if video:
        video_path = os.path.join(UPLOAD_FOLDER, video['filename'])
        if os.path.exists(video_path):
            os.remove(video_path)

        frame_folder = os.path.join(FRAMES_FOLDER, os.path.splitext(video['filename'])[0])
        if os.path.exists(frame_folder):
            for f in os.listdir(frame_folder):
                os.remove(os.path.join(frame_folder, f))
            os.rmdir(frame_folder)

    return redirect('/uploads')
@app.route('/save_video/<int:video_id>', methods=['POST'])
def save_video(video_id):
    # Implement logic for save (bookmark, export, or move)
    return redirect('/uploads')

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

def extract_frames(video_path, frame_rate, save_folder, video_id):
    cap = cv2.VideoCapture(video_path)
    count = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    interval = max(1, fps // frame_rate)
    frame_ids = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        if count % interval == 0:
            frame_filename = os.path.join(save_folder, f"frame_{count}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_id = insert_frame(video_id, frame_filename, count)
            frame_ids.append(frame_id)
        count += 1

    cap.release()
    return frame_ids

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

if __name__ == "__main__":
    app.run(debug=True)
