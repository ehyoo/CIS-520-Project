import numpy as np 
import json
'''
Before beginning, the schema of our CSV is:
label, comment, author, subreddit, score, ups, downs, date, created_utc, parent_comment
With tab delimiters.

What we want to do: 
Make sure that general summary statistics are in line with our sample
    We do this by:
    - keeping track of the ratio of subreddits
    - keeping track of totals of labels, score, ups, downs, then dividing by the total num of rows at the end
      to check that the averages of them are reasonable.

Select our sample.
    We do this by:
    - Draw a random variable from Unif[0, 1]
    - If the random variable is between 0-0.1, then select the row.
    - Otherwise, do not select the row.
    
    This should cut down our file to 3GB instead of 30GB.
'''

# First, set our seed.
np.random.seed(14)

limit = 100030
counter = 1

CUTOFF = 0.1

statistics_dict = {
    "labels": {
        "0": 0,
        "1": 0
    },
    "subreddits": {},
    "num_lines": 0,
    "agg_ups": 0,
    "agg_downs": 0,
    "agg_score": 0
}

line_cache = []

with open('../data/train-unbalanced.csv') as f:
    while True:
        line = f.readline()
        if not line:
            print("Reached end.") # Just to keep track of where we are.
            with open('../data/intermediate_summary_statistics.csv', 'w+') as note:
                json.dump(statistics_dict, note)
            with open('../data/sample_train_unbalanced.csv', 'a') as note:
                for selected_line in line_cache:
                    note.write(selected_line)
            line_cache = []
            print("no more lines")
            break

        split_line = line.split('\t')
        # Firstly, getting our label
        label = split_line[0]
        statistics_dict['labels'][label] = statistics_dict['labels'][label] + 1
        # Then, the subreddit
        subreddit = split_line[3]
        if subreddit not in statistics_dict['subreddits']:
            statistics_dict['subreddits'][subreddit] = 0
        statistics_dict['subreddits'][subreddit] = statistics_dict['subreddits'][subreddit] + 1
        # Then, the score, ups, and downs
        score = int(split_line[4])
        ups = int(split_line[5])
        downs = int(split_line[6])
        statistics_dict['agg_score'] = statistics_dict['agg_score'] + score
        statistics_dict['agg_ups'] = statistics_dict['agg_ups'] + ups
        statistics_dict['agg_downs'] = statistics_dict['agg_downs'] + downs

        # Then, get our sample
        rand_draw = np.random.uniform()
        if rand_draw <= CUTOFF:
            line_cache.append(line)

        if counter % 1000000 == 0:
            print("Reached " + str(counter) + "... Writing results to disk") # Just to keep track of where we are.
            with open('../data/intermediate_summary_statistics.csv', 'w+') as note:
                json.dump(statistics_dict, note)
            with open('../data/sample_train_unbalanced.csv', 'a') as note:
                for selected_line in line_cache:
                    note.write(selected_line)
            line_cache = []

        counter = counter + 1
