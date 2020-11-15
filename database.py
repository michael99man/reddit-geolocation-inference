import sqlite3

conn = sqlite3.connect('data/reddit_data.db')


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

# get users that have activity rows associated with them (i.e. done)
def get_completed_users():
    c = conn.cursor()

    usernames = set()
    for row in c.execute('SELECT username FROM activity GROUP BY username'):
        usernames.update([row[0]])

    print("%d users have been already processed" % len(usernames))
    return usernames

def insert_activity_entries(entries):
    c = conn.cursor()
    c.executemany('INSERT INTO activity (username, timestamp, subreddit, content, is_post) VALUES (?, ?, ?,?,?)', entries)
    conn.commit()