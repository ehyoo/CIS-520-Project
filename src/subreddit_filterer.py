'''
Filtering by subreddit
'''

# http://redditlist.com/
# Top ten by recent activity, accessed on April 15, 2019
top_ten_subreddits = [
    'askreddit',
    'news',
    'funny',
    'todayilearned',
    'gifs',
    'gaming',
    'aww',
    'pics',
    'politics',
    'askmen'
]

lines = []

counter = 0
with open('../data/test-unbalanced.csv') as f:
    while True:
        line = f.readline()
        if not line:
            print("Reached end.") # Just to keep track of where we are.
            break
        split_line = line.split('\t')
        subreddit = split_line[3]
        if subreddit.lower() in top_ten_subreddits:
            lines.append(line)
        counter = counter + 1
        if counter % 10000 == 0:
            print(counter)


with open('../data/samples/testing_top_ten_subreddits.csv', 'w+') as note:
    for line in lines:
        note.write(line + '\n')


