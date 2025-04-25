# db.py
from flask_mysqldb import MySQL
import MySQLdb.cursors

mysql = MySQL()

def init_mysql(app):
    app.config['MYSQL_HOST'] = 'localhost'
    app.config['MYSQL_USER'] = 'root'  # Replace with your MySQL user
    app.config['MYSQL_PASSWORD'] = 'edwinpaul1811'  # Replace with your MySQL password
    app.config['MYSQL_DB'] = 'cyber_eye'
    mysql.init_app(app)

def get_user_by_username(username):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
    return cursor.fetchone()

def insert_user(username, password_hash):
    try:
        cursor = mysql.connection.cursor()
        cursor.execute('INSERT INTO users (username, password) VALUES (%s, %s)', (username, password_hash))
        mysql.connection.commit()
        cursor.close()
    except MySQLdb.IntegrityError as e:
        print(f"Error: {e}")
        return False
    return True


def get_mysql():
    return mysql