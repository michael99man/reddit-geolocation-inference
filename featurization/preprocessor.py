#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Performs data cleaning on the raw database
"""

from collections import OrderedDict
from featurization import mysql
import re
import nltk
import gensim
from gensim.parsing.preprocessing import remove_stopwords


unique_words = set()

def init():
    users = mysql.fetch_users()
    cleaned_users = mysql.fetch_cleaned_users()

    cleaned_users = set(map(lambda x: x['username'], cleaned_users))
    print(cleaned_users)

    count = 0
    for user_entry in users:
        username = user_entry['username']

        # skip completed users
        if(username in cleaned_users):
            continue

        # tally activity and consolidate into arrays
        activity = mysql.fetch_activity_for_user(username)
        sub_tally, time_tally = tally(activity)

        tokens = preprocess(activity)

        unique_words.update(tokens)
        #print(user_entry)
        #print(sub_tally)
        #print(time_tally)

        mysql.store_cleaned(username, sub_tally, time_tally, tokens)

        count+=1

        # print progress
        if(count % 1000 == 0):
            print("%d/%d users processed" % (count, len(users)))
            print("%d unique words" % len(unique_words))

    mysql.store_batched()

def preprocess(activity):
    # join all their posts/comments content together
    content = ' '.join(map(lambda act: act['content'], activity))
    content = remove_stopwords(content)

    # preprocess text
    tokens = gensim.utils.simple_preprocess(content)

    unique_words.update(tokens)
    return tokens

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

    sub_tally = dict(OrderedDict(sorted(sub_tally.items())))
    # print(OrderedDict(sorted(time_tally.items())))

    time_arr = []
    for bucket in range(24*6):
        time_arr.append(time_tally.get(bucket, 0))
    return sub_tally, time_arr

# converts epoch time to which 10 minute bucket
def timestamp_to_bucket(timestamp):
    # ignores which day it is, then divides by 600 seconds
    return timestamp % 86400 // 600


if __name__ == "__main__":
    init()
