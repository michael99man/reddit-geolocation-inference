from retriever import Retriever
import pickler
import helpers
import sys
import database
import visualizations
import time

# fetch users from r/politics and state subreddits
def fetch_users(retriever):
    politics_users, states_users = pickler.read()
    # fetch 100k flaired users from r/politics
    print("Starting r/politics user scrape")
    updated_users = retriever.get_politics_users(100000, politics_users)
    print(updated_users)
    print("FETCHED %d users" % len(updated_users))

    # fetch users from each state subreddit
    retriever.get_state_subreddit_users(states_users)


# fetch activity history for all users, then write into DB
def fetch_activity(retriever):
    start_time = time.time()
    users = database.get_all_users()
    completed_users = database.get_completed_users()

    total = len(users)
    # starting point
    count = len(completed_users)

    def print_rate(count):
        users_per_min = (count-len(completed_users))*60/(time.time() - start_time)
        print("Processing at a rate of %.2f per minute" % users_per_min)

    # sort the users
    # reverse for instance 1
    process_in_reverse = (retriever.instance == 1)
    users = sorted(users, key=lambda user: user[0], reverse=process_in_reverse)

    if retriever.instance == 2:
        users = users[round(len(users)/2):]

    for user in users:
        username = user[0]
        if username in completed_users:
            print("Skipping %s" % username)
            continue

        # get all history
        try:
            entries = retriever.get_activity_history(username)
            # write history to DB
            database.insert_activity_entries(entries)
            print("Processed %d entries for user %s (%d/%d)" % (len(entries), username, count, total))
            count += 1

            if(count % 50 == 0):
                print_rate(count)
        except Exception:
            print("Failed to fetch for user %s" % username)
            continue


def main():
    instance = int(sys.argv[1])
    retriever = Retriever(instance)
    print("Initialized Retriever connections for instance %d" % instance)
    # fetch_users(retriever)
    fetch_activity(retriever)

def plotting():
    users = database.get_all_users()
    visualizations.plot_location_pie_chart(users)
    visualizations.plot_user_source(users)


if __name__ == "__main__":
    main()
