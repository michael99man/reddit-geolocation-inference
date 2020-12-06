import re

politics_flair_pattern = re.compile(':.*: ')
bad_flairs = ["I voted", "PolitiFact", "Iowa Caucus", "Antarctica", "Verified", "Samuel Doctor", "The Moon", "Guam", "Cartoonist", "Washington Post", "Virgin Islands", "Clayburn Griffin", "Puerto Rico"]

USERNAME_QUEUE = "usernames"
ACTIVITY_QUEUE = "activity"

# cleans up the flairs from r/politics
def clean_politics_flair(flair):
    for bad in bad_flairs:
        if bad in flair:
            return None
        if flair == "Texas!":
            flair = "Texas"

    # remove the flag emoji
    sub = politics_flair_pattern.sub('', flair)
    if(len(sub) < 3):
        return None
    else:
        return sub


def print_politics_flairs(users):
    flairs = set()
    for s in users.values():
        flairs.update([s])
    print(flairs)

    for state in states:
        if state not in flairs:
            print("Missing %s" % state)