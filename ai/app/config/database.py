# app/config/database.py
import os
import pymysql
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    return pymysql.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", "1234"),
        database=os.getenv("DB_NAME", "my_project_db"),
        charset="utf8mb4",
        # cursorclass=pymysql.cursors.DictCursor   # ★ 핵심 포인트
    )
    return conn

def fetch_all(query, params=None):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
    conn.close()
    return rows