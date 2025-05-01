-- Create Database
CREATE DATABASE IF NOT EXISTS cyber_eye;
USE cyber_eye;

-- Create USER table
CREATE TABLE user (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login DATETIME NULL
);

-- Create VIDEO table
CREATE TABLE video (
    video_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(255) NOT NULL,
    fps INT NOT NULL,
    upload_date DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    duration INT NOT NULL,
    format VARCHAR(20) NOT NULL,
    FOREIGN KEY (user_id) REFERENCES user(user_id)
        ON DELETE CASCADE
);

-- Create FRAME table
CREATE TABLE frame (
    frame_id INT AUTO_INCREMENT PRIMARY KEY,
    video_id INT,
    frame_path VARCHAR(255) NOT NULL,
    frame_number INT NOT NULL,
    extraction_date DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES video(video_id)
        ON DELETE CASCADE
);

-- Create FACE table
CREATE TABLE face (
    face_id INT AUTO_INCREMENT PRIMARY KEY,
    frame_id INT,
    x_position FLOAT NOT NULL,
    y_position FLOAT NOT NULL,
    width FLOAT NOT NULL,
    height FLOAT NOT NULL,
    confidence FLOAT NOT NULL,
    FOREIGN KEY (frame_id) REFERENCES frame(frame_id)
        ON DELETE CASCADE
);

-- Create ANALYSIS_RESULT table
CREATE TABLE analysis_result (
    result_id INT AUTO_INCREMENT PRIMARY KEY,
    face_id INT,
    deepfake_probability FLOAT NOT NULL,
    analysis_date DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(50) NOT NULL,
    FOREIGN KEY (face_id) REFERENCES face(face_id)
        ON DELETE CASCADE
);

-- Create VIDEO_RESULT table
CREATE TABLE video_result (
    video_result_id INT AUTO_INCREMENT PRIMARY KEY,
    video_id INT,
    overall_deepfake_score FLOAT NOT NULL,
    total_frames_analyzed INT NOT NULL,
    total_faces_detected INT NOT NULL,
    analysis_complete_date DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES video(video_id)
        ON DELETE CASCADE
);
