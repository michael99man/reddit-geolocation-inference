""" Provides a simplified interface to the MySQL DB
"""

import pymysql.cursors

# Connect to the database
connection = pymysql.connect(host='localhost',
                             user='root',
                             password='[redacted]',
                             db='reddit_data',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)
entry_batch  = []

def fetch_users():
    with connection.cursor() as cursor:
        sql = "SELECT * FROM `users`"
        cursor.execute(sql)
        rows = cursor.fetchall()
        return rows

def fetch_activity_for_user(username):
    with connection.cursor() as cursor:
        sql = "SELECT * FROM `activity` where username=%s"
        cursor.execute(sql, (username,))
        rows = cursor.fetchall()
        return rows


def fetch_all_cleaned(limit = None):
    with connection.cursor() as cursor:
        sql = "SELECT * FROM `cleaned`s"
        if (limit is not None):
            sql = "SELECT * FROM `cleaned`s LIMIT %d" % limit;
        cursor.execute(sql)
        rows = cursor.fetchall()
        return rows

def fetch_cleaned(username):
    with connection.cursor() as cursor:
        sql = "SELECT * FROM `cleaned` where username=%s"
        cursor.execute(sql, (username,))
        rows = cursor.fetchone()
        return rows

def fetch_cleaned_users():
    with connection.cursor() as cursor:
        sql = "SELECT username FROM `cleaned`"
        cursor.execute(sql)
        rows = cursor.fetchall()
        return rows

def store_cleaned(username, sub_tally, time_tally, tokens):
    global entry_batch
    entry_batch.append((username, str(sub_tally), str(time_tally), str(tokens)))

    if(len(entry_batch) >= 100):
        store_batched()

def store_batched():
    global entry_batch
    if len(entry_batch) > 0:
        print(
            "Committing %d entries to database, users (%s->%s)" % (len(entry_batch), entry_batch[0][0], entry_batch[-1][0]))
        with connection.cursor() as cursor:
            sql = "INSERT IGNORE INTO cleaned (username, subreddit_arr, timestamp_arr, document) VALUES (%s, %s, %s, %s)"
            cursor.executemany(sql, entry_batch)
            connection.commit()
            entry_batch = []
