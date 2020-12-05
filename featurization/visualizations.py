
import matplotlib.pyplot as plt
import helpers
import pickle

def plot_location_pie_chart(users):
    states = {"Foreign": 0}
    for s in helpers.states:
        states[s] = 0

    for user in users:
        location = user[1]
        if(location in states):
            states[location] +=1
        else:
            print("Foreign %s" % location)
            states["Foreign"] +=1



    fig1, ax1 = plt.subplots()
    ax1.pie(states.values(), labels=states.keys(), autopct='%1.1f%%', shadow=False, startangle=90, textprops={'fontsize': 6})
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    fig1 = plt.gcf()
    fig1.set_size_inches(10, 10)
    circle = plt.Circle(xy=(0, 0), radius=0.75, facecolor='white')
    plt.gca().add_artist(circle)
    plt.show()

def plot_user_source(users):
    sources = {'r/politics Flairs':0, 'State-based subreddits':0}

    for user in users:
        if(user[2] == 1):
            sources['r/politics Flairs'] +=1
        elif user[2] == 0:
            sources['State-based subreddits'] +=1
        else:
            print("Wtf")

    fig1, ax1 = plt.subplots()
    ax1.pie(sources.values(), labels=sources.keys(), autopct='%1.1f%%', shadow=False, startangle=90,
            textprops={'fontsize': 14})
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    fig1 = plt.gcf()
    fig1.set_size_inches(10, 10)
    circle = plt.Circle(xy=(0, 0), radius=0.75, facecolor='white')
    plt.gca().add_artist(circle)
    plt.show()



# plots a histogram of how many activity entries
def plot_user_activity_counts():
    rows = pickle.load(open("data/unique_users.p", "rb"))
    print(rows)
    print(len(rows))

    count = 0
    for row in rows:
        if row[1] >=500:
            count += 1
    print(count)

    vals = list(map(lambda row: row[1], rows))
    n_bins = 20

    fig, ax = plt.subplots(1,1)
    ax.hist(vals, n_bins, range=(0, 3000))
    ax.set_title("Histogram")
    plt.show()