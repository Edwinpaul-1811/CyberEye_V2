# db.py
from flask_mysqldb import MySQL
import MySQLdb.cursors

mysql = MySQL()

def init_mysql(app):
    app.config['MYSQL_HOST'] = 'localhost'
    app.config['MYSQL_USER'] = 'root' # Replace with your MySQL username
    app.config['MYSQL_PASSWORD'] = 'edwinpaul1811' # Replace with your MySQL password
    app.config['MYSQL_DB'] = 'cyber_eye'
    mysql.init_app(app)

def get_videos_by_user(user_id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT * FROM video WHERE user_id = %s', (user_id,))
    videos = cursor.fetchall()
    cursor.close()
    return videos

def get_video_filename(video_id, user_id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT filename FROM video WHERE video_id = %s AND user_id = %s', (video_id, user_id))
    video = cursor.fetchone()
    cursor.close()
    return video


def delete_video_by_id(video_id, user_id):
    cursor = mysql.connection.cursor()
    cursor.execute('DELETE FROM video WHERE video_id = %s AND user_id = %s', (video_id, user_id))
    mysql.connection.commit()
    cursor.close()

def get_user_by_identifier(identifier):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT * FROM user WHERE username = %s OR email = %s', (identifier, identifier))
    return cursor.fetchone()

def insert_user(username, email, password_hash):
    try:
        cursor = mysql.connection.cursor()
        cursor.execute('INSERT INTO user (username, email, password_hash) VALUES (%s, %s, %s)', (username, email, password_hash))
        mysql.connection.commit()
        cursor.close()
    except MySQLdb.IntegrityError as e:
        print(f"Error: {e}")
        return False
    return True

def insert_video(user_id, filename, file_path, fps, duration, format):
    cursor = mysql.connection.cursor()
    cursor.execute('''
        INSERT INTO video (user_id, filename, file_path, fps, duration, format)
        VALUES (%s, %s, %s, %s, %s, %s)
    ''', (user_id, filename, file_path, fps, duration, format))
    mysql.connection.commit()
    video_id = cursor.lastrowid
    cursor.close()
    return video_id

def insert_frame(video_id, frame_path, frame_number):
    cursor = mysql.connection.cursor()
    cursor.execute('''
        INSERT INTO frame (video_id, frame_path, frame_number)
        VALUES (%s, %s, %s)
    ''', (video_id, frame_path, frame_number))
    mysql.connection.commit()
    frame_id = cursor.lastrowid
    cursor.close()
    return frame_id

def insert_face(frame_id, x, y, width, height, confidence):
    cursor = mysql.connection.cursor()
    cursor.execute('''
        INSERT INTO face (frame_id, x_position, y_position, width, height, confidence)
        VALUES (%s, %s, %s, %s, %s, %s)
    ''', (frame_id, x, y, width, height, confidence))
    mysql.connection.commit()
    face_id = cursor.lastrowid
    cursor.close()
    return face_id

def insert_analysis_result(face_id, deepfake_probability, model_version="v1.0"):
    cursor = mysql.connection.cursor()
    cursor.execute('''
        INSERT INTO analysis_result (face_id, deepfake_probability, model_version)
        VALUES (%s, %s, %s)
    ''', (face_id, deepfake_probability, model_version))
    mysql.connection.commit()
    cursor.close()

def insert_video_result(video_id, overall_score, total_frames, total_faces):
    cursor = mysql.connection.cursor()
    cursor.execute('''
        INSERT INTO video_result (video_id, overall_deepfake_score, total_frames_analyzed, total_faces_detected)
        VALUES (%s, %s, %s, %s)
    ''', (video_id, overall_score, total_frames, total_faces))
    mysql.connection.commit()
    cursor.close()

def get_mysql():
    return mysql
