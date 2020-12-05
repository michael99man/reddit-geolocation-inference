#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Performs data cleaning on the raw database
"""
import pymysql.cursors
import helpers
from collections import OrderedDict


# Connect to the database
connection = pymysql.connect(host='localhost',
                             user='root',
                             password='Toatoa123!m',
                             db='reddit',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

def init():
    users = fetch_users()

    for user_entry in users:
        username = user_entry['username']
        print(user_entry)
        activity = fetch_activity_for_user(username)
        print(activity)
        print(len(activity))
        sub_tally, time_tally = tally(activity)
        print(sub_tally)
        print(time_tally)
        return

# tallies the subreddits and the timestamps
def tally(activity):
    sub_tally = dict()
    time_tally = dict()

    for entry in activity:
        subreddit = entry['subreddit']
        count = sub_tally.get(subreddit, 0)
        sub_tally[subreddit] = count+1

        bucket = timestamp_to_bucket(entry['timestamp'])
        time_tally[bucket] = time_tally.get(bucket, 0) + 1

    sub_tally = OrderedDict(sorted(sub_tally.items()))
    time_tally = OrderedDict(sorted(time_tally.items()))

    return sub_tally, time_tally

# converts epoch time to which 10 minute bucket
def timestamp_to_bucket(timestamp):
    # ignores which day it is, then divides by 600 seconds
    return timestamp % 86400 // 600


def fetch_users():
    with connection.cursor() as cursor:
        sql = "SELECT * FROM `users`"
        cursor.execute(sql)
        rows = cursor.fetchall()
        print(rows)
        return rows

def fetch_activity_for_user(username):
    with connection.cursor() as cursor:
        sql = "SELECT * FROM `activity` where username=%s"
        cursor.execute(sql, (username,))
        rows = cursor.fetchall()
        print(rows)
        return rows

if __name__ == "__main__":
    init()
