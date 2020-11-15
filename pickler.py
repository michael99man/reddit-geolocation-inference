import pickle

def read():
    try:
        politics_users = pickle.load(open("data/politics_users.p", "rb"))
    except FileNotFoundError:
        politics_users = {}
    try:
        states_users = pickle.load(open("data/states_users.p", "rb"))
    except FileNotFoundError:
        states_users = {}

    print("Loaded %d pickled politics users, %d pickled states users" % (len(politics_users), len(states_users)))
    return (politics_users, states_users)


def write_politics(users):
    pickle.dump(users, open("data/politics_users.p", "wb"))
    print("Saved %d pickled politics users" % len(users))

def write_states(users):
    pickle.dump(users, open("data/states_users.p", "wb"))
    print("Saved %d pickled states users" % len(users))