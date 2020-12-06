#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Handles interfacing with Sqlite3 database
"""

import sqlite3
import pickle

conn = sqlite3.connect('data.nosync/reddit_data.db')

def insert_users(politics_users, states_users):
    # convert dictionaries into lists, with the int representing where the user came from
    users = []
    for (username, location) in politics_users.items():
        users.append((username, location, 1))

    overlap = 0
    mismatches = 0
    for (username, location) in states_users.items():
        # defer to r/politics flairs as source of truth
        if username in politics_users:
            # print("OVERLAP %s" %username)
            if location != politics_users[username]:
                print("MISMATCH: %s (%s, %s)" % (username, politics_users[username], location))
                mismatches += 1
            overlap += 1
            continue
        users.append((username, location, 0))

    print("%d overlaps found" % overlap)
    print("%d mismatches found" % mismatches)

    c = conn.cursor()
    c.executemany('INSERT INTO users (username, location, is_from_politics) VALUES (?, ?, ?)', users)

    for row in c.execute('SELECT * FROM users'):
        print(row)
    conn.commit()

# load the contents of the users table into memory
def get_all_users():
    c = conn.cursor()
    c.execute('SELECT * FROM users')
    rows = c.fetchall()

    print("Loaded %d users from DB" % len(rows))
    return rows

def insert_activity_entries(entries):
    c = conn.cursor()
    c.executemany('INSERT INTO activity (username, timestamp, subreddit, content, is_post) VALUES (?, ?, ?,?,?)', entries)
    conn.commit()


# get the set of users that have activity rows associated with them (i.e. done)
def get_completed_users():
    c = conn.cursor()
    c.execute('SELECT DISTINCT username FROM activity')
    rows = c.fetchall()

    completed_users = set()
    for row in rows:
        completed_users.update([row[0]])

    print("%d users have been already processed" % len(completed_users))
    return completed_users


# delete all users that don't have activity associated with them (because of errors, privacy settings)
def delete_null_users():
    all_users = get_all_users()
    completed_users = get_completed_users()

    null_users = []

    for row in all_users:
        username = row[0]
        if username not in completed_users:
            null_users.append(username)

    null_users_str = str(null_users)
    null_users_str = null_users_str.replace("[", "(").replace("]", ")")
    print("Deleting %d users: %s" % (len(null_users), null_users_str))

    c = conn.cursor()
    c.execute('DELETE FROM users where username in ' + null_users_str)
    conn.commit()


# delete users that have activity counts outside of the range [500,3000]
def delete_low_count_users():
    rows = pickle.load(open("data/unique_users.p", "rb"))

    bad_users = []
    for row in rows:
        if row[1] < 500 or row[1] > 3000:
            bad_users.append(row[0])

    bad_users_str = str(bad_users).replace("[", "(").replace("]", ")")
    print(len(bad_users))
    print(bad_users_str)

    c = conn.cursor()
    c.execute('DELETE FROM users where username in ' + bad_users_str)
    conn.commit()



# gets the distinct users with their activity counts
def get_activity_counts():
    c = conn.cursor()
    c.execute("SELECT username, COUNT(*) FROM activity GROUP BY username")
    rows = c.fetchall()
    pickle.dump(rows, open("data/unique_users.p", "wb"))


# gets all activity entries associated with a username
def get_activity_for_user(username):
    c = conn.cursor()
    c.execute("SELECT * FROM activity WHERE username='" + username +"'")
    rows = c.fetchall()
    print("Fetched %d rows for %s" % (len(rows), username))
    return rows