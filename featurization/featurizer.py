""" Computes the full feature vectors based on the time_model, word2vec model, and subreddits
"""

import pymysql
from collections import defaultdict
import json
import numpy as np
from featurization import mysql


# takes in a subreddit_arr and converts it to the normalized indexed form
# creates a sparse vector of subreddit activity
def normalized_subreddit_vector(subreddit_arr):
    v = np.zeros(sub_n)

    for subreddit in subreddit_arr:
        if(subreddit in sub_dict):
            v[sub_dict[subreddit]] +=1

    norm = np.linalg.norm(v)
    if norm == 0:
        return v

    return v / np.linalg.norm(v)

def tokenizer(arr_string):
    return json.loads(arr_string.replace("\'", "\""))

def load_data():
    # fetch users and cleaned tables
    cleaned_rows = mysql.fetch_all_cleaned();
    users_rows = mysql.fetch_users();

    # map of user to (label, features) tuples
    data = dict()

    # load users
    for r in users_rows:
        data[r['username']] = {'location': r['location'], 'is_from_politics': r['is_from_politics']}


    # count # of subreddits
    subreddits = defaultdict(int)

    for r in cleaned_rows:
        username = r['username']
        #print(r)
        arr_str = r['subreddit_arr']
        subreddit_arr = json.loads(arr_str.replace("\'", "\""))
        for s in subreddit_arr:
            subreddits[s] += 1

    print("Read %d unique subreddits" % len(subreddits))

    # sort most popular subreddits
    sorted_subs = sorted(subreddits.items(), key=lambda v: v[1])
    sorted_subs.reverse()


    # remove 300 most popular subreddits and keep next top 5000 subreddits
    discard_n = 300
    sub_n = 5000
    sub_5k = sorted_subs[discard_n:discard_n+sub_n]
    sub_dict = dict([(sub_5k[i][0], i) for i in range(len(sub_5k))])

    print("Kept a dict of the top %d->%d subreddits" % (discard_n, sub_n))

    # create subreddit vectors for each user
    count = 0
    for row in cleaned_rows:
        username = row['username']
        subreddits = tokenizer(row['subreddit_arr'])
        sub_v = normalized_subreddit_vector(subreddits)
        data[username]['subreddit_v'] = sub_v
        count +=1

        if(count % 10000 == 0):
            print("Processed %d entries" % count)

    return data
