#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Helper functions to interact with the Reddit API through PRAW and PSAW
"""

import praw
from praw.models import MoreComments
from psaw import PushshiftAPI
from collection.helpers import states
from collection import pickler, helpers

ids = ["pw9MViKdnvWhvg", "NS3agxMZ4b-jsg", "9LMNOYIf3-aHcg", "25ziqrlE-oLLaw", "Z9o5Pqo2wb3DOQ", "eimkyYu2Ilur0Q", "0Xr2T0wag5tpzg", "qyxztQqeBQS5sA", "roBbLMyO9IePSA", "RvUqLB0PyFPeTw", "_f_atr1DIqcnNw","Q1Oy8MK_h-3FYg", "qXHeySUMAL4zzw", "2JfJq5rziYDE0w","gerQxFBoNoECjA","VaU5LAmEjCXtnw","EbAxC0O6FlZZ6w","cKlKIpDcc6SWjQ","40aYBMCM1KxqQw", "y4b76WQNiDa6Lg"]
secrets = ["N3DpHmI8cRJ1vUZQmCyjrj3_KSAcZg", "4A-2iK0DNfsRo0ZuYpMxtkAqRQ9rGQ", "zn84HS4knQZxJqQq0lAkSnvXCwT0vw", "K_FN3Hh8BDthyFtCjU6x1ITX9inXLQ", "jUEDw8bLgqAbu3eZcXaCamdnYvmFRQ", "vJZyiZ0KDWoM-uCvOLrlT6fUL4SBfg", "isOyyTg_Z0gWaVPb78FUcJKZlamvbg", "GUpfdN9DWjYp_0ri6soT_gU9fwGOKA", "CGh8Ue8MgHkwnw8YyeDEcmz2SZ3itQ", "gOvnjiUThZXK9GaZyfJofFEwaLo2bg", "nvH5jgeLxj2k10-LdIphUSrVzCvV4w","ErDsETn7CwOzmCaRShdz3PL9EkDwHA","tL661goKFwcy0_t6hqBj47pvJjJ0GQ","1J1ICzRp8Tqqu2wyZoX_dLfzXsQiPg","PZOwoIo183R0XuDVBXWwSTSDdKOYKQ","A3Q7tQ3aNXvj_bfL-epYKFVxL_schw","jqXe2JJ3QEw6yvsbLx1h-msl0XbPdg","kalud-IZs6mqFg6hbquc4yKW_zoISQ","7js4pFgg9qaWBMloMkATKzFGiGvyNw", "GGRJLEtCcsTgl0vBqSJH1OfwlvJJhg"]
usernames = ["ele574-scraper", "ele574-scraper1", "ele574-scraper2", "ele574-scraper3", "ele574-scraper4", "ele574-scraper5", "ele574-scraper6", "ele574-scraper7", "ele574-scraper8", "ele574-scraper9", "ele574-scraper10", "ele574-scraper11", "ele574-scraper12", "ele574-scraper13", "ele574-scraper14", "ele574-scraper15", "ele574-scraper16", "ele574-scraper17", "ele574-scraper18", "ele574-scraper19"]

class Retriever:
    def __init__(self, instance):
        self.reddit = praw.Reddit(
            client_id=ids[instance],
            client_secret=secrets[instance],
            user_agent="script:net.michaelman:v0.1 (by u/ele574-scraper)",
            username= usernames[instance],
        )
        self.instance = instance
        self.pushshift = PushshiftAPI()

    # get `num_users` flaired users from r/politics
    def get_politics_users_PUSHSHIFT(self, num_users, users={}):
        end_epoch = 10000000000
        # keep looping through entries
        while True:
            entries = list(self.pushshift.search_comments(before=end_epoch,
                                                          subreddit='politics',
                                                          filter=['author', 'author_flair_text'],
                                                          limit=10000))
            print("Fetched %d entries before %d" % (len(entries), end_epoch))
            for e in entries:
                # the created epoch should be decreasing on each entry
                end_epoch = e.created_utc
                flair = helpers.clean_politics_flair(e.author_flair_text)
                username = e.author
                if (username not in users and flair is not None):
                    users[username] = flair
                    print("%d | %s: %s" % (len(users), username, flair))
                    if (len(users) == num_users):
                        return users

    # get `num_users` flaired users from r/politics
    def get_politics_users(self, num_users, users={}):
        politics_sub = self.reddit.subreddit("politics")
        count = 0

        # iterate through the top posts on r/politics
        for post in politics_sub.top("all"):
            # if post.url == "https://www.reddit.com/r/politics/comments/jptq5n/megathread_joe_biden_projected_to_defeat/":
            #  continue
            print("Post %d: %s" % (count, post.url))
            count += 1
            comments = post.comments
            before = len(post.comments)
            comments.replace_more(limit=None, threshold=10)
            after = len(post.comments)

            print("Fetched (%d->%d) comments" % (before, after))

            for comment in comments:
                # skip
                if isinstance(comment, MoreComments):
                    continue

                # get the flair and clean it up into a state/country name
                flair = helpers.clean_politics_flair(comment.author_flair_text)

                if flair is not None:
                    username = comment.author.name
                    if (username not in users):
                        users[username] = flair
                        print("%d | %s: %s" % (len(users), username, flair))

                        # save every 1000 users
                        if (len(users) % 1000 == 0):
                            pickler.write_politics(users)

                        # stop accumulating at `num_users` unique users
                        if (len(users) >= num_users):
                            pickler.write_politics(users)
                            return users
        # prematurely finished loading
        pickler.write_politics(users)
        return users

    # gets users from each state subreddit
    def get_state_subreddit_users(self, users={}):
        print("%d states" % len(states))
        completed_states = set()
        for s in users.values():
            completed_states.update([s])
        print("Completed states: %s" % completed_states)

        for state in states:
            if state == "District of Columbia":
                sub_name = "WashingtonDC"
            else:
                sub_name = state.replace(" ", "")
            print("Processing: %s at r/%s" % (state, sub_name))

            # check if state has been processed
            if state in completed_states:
                print("Already processed %s, skipping...." % state)
                continue

            # for some reason this only fetches 5000 comments
            comments = list(self.pushshift.search_comments(subreddit=sub_name,
                                                           filter=['author', 'author_flair_text'],
                                                           limit=100000))
            count = 0
            add_count = 0
            for comment in comments:
                count += 1
                if comment.author not in users:
                    users[comment.author] = state
                    add_count += 1

            print("%d comments for r/%s, %d users added" % (count, sub_name, add_count))

            # save results
            pickler.write_states(users)

    # returns activity as a list of entries from Pushshift
    # each entry will be of the form (subreddit, is_post (boolean), timestamp, content)
    def get_activity_history_PUSHSHIFT(self, username):
        entries = []

        # fetch all posts from this user
        submissions = self.pushshift.search_submissions(author=username,
                                                        filter=['subreddit', 'created_utc', 'title', 'selftext'],
                                                        limit=50000)
        for submission in submissions:
            if not hasattr(submission, "selftext"):
                # fallback to title if no body
                content = submission.title
            else:
                content = submission.selftext
            timestamp = round(submission.created)

            entry = (username, timestamp, submission.subreddit, content, 1)
            entries.append(entry)

        # fetch all comments from this user
        comments = self.pushshift.search_comments(author=username, filter=['subreddit', 'created_utc', 'body'],
                                                  limit=50000)

        for comment in comments:
            content = comment.body
            timestamp = round(comment.created)

            entry = (username, timestamp, comment.subreddit, content, 0)
            entries.append(entry)
        return entries

    # returns activity as a list of entries from Reddit API
    # each entry will be of the form (username, subreddit, is_post (boolean), timestamp, content)
    def get_activity_history(self, username):
        entries = []

        # fetch all posts from this user
        submissions = self.reddit.redditor(username).submissions.top("all", limit=None)

        for submission in submissions:
            if submission.selftext != "":
                content = submission.selftext
            else:
                # fallback to title if no body
                content = submission.title
            timestamp = round(submission.created_utc)

            entry = (username, timestamp, submission.subreddit.display_name, content, 1)
            entries.append(entry)

        # fetch all comments from this user
        comments = self.reddit.redditor(username).comments.top("all", limit=None)
        comment_ids = set()
        count = 0
        for comment in comments:
            content = comment.body
            timestamp = round(comment.created_utc)

            entry = (username, timestamp, comment.subreddit.display_name, content, 0)
            entries.append(entry)
            comment_ids.update([comment.id])
            count += 1

        if count == 1000:
            # fetch most controversial comments to increase total comment count
            comments = self.reddit.redditor(username).comments.controversial("all", limit=None)
            for comment in comments:
                if comment.id in comment_ids:
                    continue
                content = comment.body
                timestamp = round(comment.created_utc)

                entry = (username, timestamp, comment.subreddit.display_name, content, 0)
                entries.append(entry)
                comment_ids.update([comment.id])

        return entries
