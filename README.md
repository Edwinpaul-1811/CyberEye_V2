# 🧠 CyberEye - Deepfake Detection App

CyberEye is a smart web application that detects deepfake content in uploaded videos by analyzing facial landmarks and their consistency using deep learning and clustering techniques. Built with Flask and OpenCV, it offers a user-friendly interface to upload videos, extract frames, analyze faces, and display a deepfake classification summary.

---

## 🚀 Features

- 🔒 User registration and login (username or email)
- 📹 Upload and process videos
- 🧠 Deepfake detection using landmark consistency
- 🖼 View extracted frames with color-coded results:
  - 🟢 Genuine
  - 🔴 Deepfake
- 📊 View detection summary
- 📁 Manage uploaded videos
- 📘 Educational blog on deepfakes

---

## 🛠 Technologies Used

- **Flask** (Python backend)
- **MySQL** (Database)
- **OpenCV** & **Dlib** (Image processing)
- **Bootstrap 5** (Frontend styling)
- **Scikit-learn** (Clustering for deepfake analysis)

---

## 📂 Project Structure

```
CyberEye/
│
├── static/              # Static files (optional)
├── templates/           # HTML templates (index.html, login.html, etc.)
├── uploads/             # Uploaded video files
├── frames/              # Extracted frame folders
├── model/               # Dlib predictor (.dat file)
├── app.py               # Flask app
├── db.py                # MySQL DB functions
├── predict.py           # Deepfake detection logic
├── schema.sql           # MySQL schema for tables
├── requirements.txt     # Python dependencies
└── README.md            # You're here!
```

---

## ⚙️ Installation Guide

### 1. 📦 Clone the Repository

```bash
git clone https://github.com/your-username/cybereye.git
cd cybereye
```

### 2. 🐍 Create & Activate Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. 📥 Install Dependencies

```bash
pip install -r requirements.txt
```

> Ensure you also install OpenCV and dlib system dependencies (see below).

### 4. 🏃‍♂️‍➡️ Run the Application
```bash
python app.py
```

---

## 🧰 Dlib & OpenCV Setup

**Linux/macOS**
```bash
sudo apt-get install cmake
pip install dlib opencv-python
```

**Windows**
- Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- Then install:
```bash
pip install dlib==19.24.0
pip install opencv-python
```

---

## 🧠 Dlib Landmark Model

Download the required model manually:

```bash
mkdir model
cd model
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```

---

## 🗄️ MySQL Setup

### 1. Login to MySQL and create DB

```sql
CREATE DATABASE cyber_eye;
```

### 2. Run the SQL schema (optional)

Use the provided `schema.sql` to create all 6 required tables:

```sql
-- user, video, frame, face, analysis_result, video_result
```

### 3. Update `db.py` with your credentials:
```python
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'your_mysql_password'
```

---

## ▶️ Running the App

```bash
python app.py
```

Visit: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🔐 Default Routes

| Route         | Description                            |
|---------------|----------------------------------------|
| `/`           | Redirect to login / index              |
| `/register`   | Register a new user                    |
| `/login`      | Login with username or email           |
| `/index`      | Upload video + deepfake detection      |
| `/uploads`    | Manage uploaded videos                 |
| `/blog`       | Read about deepfakes                   |
| `/logout`     | End user session                       |

---

## 📌 Notes

- All uploaded videos are saved in `/uploads/`
- Frames are saved under `/frames/{video_name}/`
- MySQL stores metadata; no video is deleted from DB, only removed from disk

---

## 🧪 Sample Use Case

1. Register an account  
2. Upload a video  
3. Frames are extracted and analyzed  
4. Deepfake probability is estimated  
5. Results shown with colored frame borders  
6. Manage uploads or read the blog to learn more

---

## ❤️ Acknowledgments

- [dlib](http://dlib.net/) for facial landmark detection  
- [OpenCV](https://opencv.org/) for frame processing  
- [Scikit-learn](https://scikit-learn.org/) for clustering  
- Bootstrap for UI styling

---
