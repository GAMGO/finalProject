import pymysql

def get_conn():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="1234",
        database="my_project_db",
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )

def get_reviews_by_store(store_id):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT review_text, rating
            FROM store_reviews
            WHERE store_idx=%s AND is_blocked=0
        """, (store_id,))
        rows = cur.fetchall()
    conn.close()
    return rows