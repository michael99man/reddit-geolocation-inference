from retriever import Retriever
import pickler
import helpers
import sys
import database
import visualizations
import time

# fetch users from r/politics and state subreddits
# OBSOLETE, REWRITE INTO WORKERS IF NEEDED
def fetch_users(retriever):
    politics_users, states_users = pickler.read()
    # fetch 100k flaired users from r/politics
    print("Starting r/politics user scrape")
    updated_users = retriever.get_politics_users(100000, politics_users)
    print(updated_users)
    print("FETCHED %d users" % len(updated_users))

    # fetch users from each state subreddit
    retriever.get_state_subreddit_users(states_users)



def main():

    database.get_activity_counts()

def plotting():
    users = database.get_all_users()
    visualizations.plot_location_pie_chart(users)
    visualizations.plot_user_source(users)


if __name__ == "__main__":
    main()
