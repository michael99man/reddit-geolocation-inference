""" Computes the full feature vectors based on the time_model, word2vec model, and subreddits
"""
import pymysql
from collections import defaultdict
import json
import numpy as np
import mysql
import pickle
from gensim.models import KeyedVectors
from torch.utils.data import Dataset

states = ["Alaska", "Alabama", "Arkansas", "Arizona", "California", "Colorado", "Connecticut", "District of Columbia", "Delaware", "Florida", "Georgia", "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]

state_subs = ["alaska", "Alabama", "Arkansas", "arizona", "California", "Colorado", "Connecticut", "washingtondc", "Delaware", "florida", "Georgia", "Hawaii", "Iowa", "Idaho", "illinois", "Indiana", "kansas", "Kentucky", "Louisiana", "massachusetts", "maryland", "Maine", "Michigan", "minnesota", "missouri", "mississippi", "Montana", "NorthCarolina", "northdakota", "Nebraska", "newhampshire", "newjersey", "NewMexico", "Nevada", "newyork", "Ohio", "oklahoma", "oregon", "Pennsylvania", "RhodeIsland", "southcarolina", "SouthDakota", "Tennessee", "texas", "Utah", "Virginia", "vermont", "Washington", "wisconsin", "WestVirginia", "wyoming"]

discard_n = 300
sub_n = 5000
# takes in a subreddit_arr and converts it to the normalized indexed form
# creates a sparse vector of subreddit activity
def normalized_subreddit_vector(subreddit_arr, sub_dict):
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

# converts each document to tokens and then returns a list of indices
def process_document(w2v, doc):
    indices = []
    for token in tokenizer(doc):
        if token in w2v.wv.vocab:
            indices.append(w2v.wv.vocab.get(token).index)
    
    return indices


# generate dictionary of subreddit -> index
def gen_sub_dict_worker(indices, cleaned_rows, tally_dict):
    for i in indices:  
        row = cleaned_rows[i]
        subreddit_arr = tokenizer(row['subreddit_arr'])
        for s in subreddit_arr:
            tally_dict[s] += 1
        if(i % 10000 == 0):
            print("Done processing %d rows" % i)
    print("Worker has completed range: %d-%d" % (indices[0], indices[-1]))


def split(a, n):
    k, m = divmod(len(a), n)
    return list((a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)))

def gen_sub_dict():
    cleaned_rows = mysql.fetch_all_cleaned();
    print("Fetched cleaned rows")
    
    # count # of subreddits
    tally = defaultdict(int)
        
    # don't parallelize
    #n_workers = 16
    #parts = split(range(len(cleaned_rows)), n_workers)
    #with ProcessPoolExecutor(max_workers=n_workers) as e:
        #for i in range(n_workers):
         #   e.submit(gen_sub_dict_worker, parts[i], cleaned_rows, tally)

    gen_sub_dict_worker(range(len(cleaned_rows)), cleaned_rows, tally)
    print("Read %d unique subreddits" % len(tally))

    # sort most popular subreddits
    sorted_subs = sorted(tally.items(), key=lambda v: v[1])
    sorted_subs.reverse()
    
    # remove 300 most popular subreddits and keep next top 5000 subreddits
    sub_5k = sorted_subs[discard_n:discard_n+sub_n]
    sub_dict = dict([(sub_5k[i][0], i) for i in range(len(sub_5k))])
    
    print("Kept a dict of the top %d->%d subreddits" % (discard_n, sub_n+discard_n))
    pickle.dump(sub_dict, open("sub_dict.p", "wb"))
    return sub_dict
    
def load_data():
    # load w2v model
    w2v = KeyedVectors.load('../models/full.model')
    
    # fetch users and cleaned tables
    cleaned_rows = mysql.fetch_all_cleaned();
    users_rows = mysql.fetch_users();

    # map of user to (label, features) tuples
    data = dict()
    
    # load time vectors
    time_vectors = pickle.load(open("../models/prob_vectors.time", 'rb'))
    sub_dict = pickle.load(open("sub_dict.p", "rb"))
    
    # load username list
    usernames = []
    for r in users_rows:
        usernames.append(r['username'])
    
    
    for r in users_rows:        
        data[r['username']] = {'location': r['location'], 'is_from_politics': r['is_from_politics']}
        
    count = 0
    for r in cleaned_rows:
        username = r['username']
        if username == '[deleted]':
            continue
            
        data[username]['document'] = process_document(w2v, r['document'])
        subreddits = tokenizer(r['subreddit_arr'])
        sub_v = normalized_subreddit_vector(subreddits, sub_dict)
        data[username]['subreddit_v'] = sub_v
        
        if username in time_vectors:
            data[username]['time_v'] = time_vectors[username]
        
        count+=1
        if(count % 1000 == 0):
            print("Processed %d entries" % count)
            
    return data

def write_chungus():
    # lists that will eventually be turned into tensors and into the dataset
    labels = []
    words = []
    subs = []
    times = []
    
    
class ChungusSet(Dataset):
    def __init__(self, words, subs, times, labels):
        self.words = words
        self.subs = subs
        self.times = times
        self.labels = labels
        assert(len(words)== len(subs) and len(subs) == len(times) and len(times) == len(labels))
    
    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        #get images and labels here 
        #returned images must be tensor
        #labels should be int 
        return self.words[idx], self.subs[idx] , self.times[idx], self.labels[idx] 
        
if __name__ == "__main__":
    gen_sub_dict()