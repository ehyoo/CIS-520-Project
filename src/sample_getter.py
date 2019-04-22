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

subreddits_dict = {}

line_cache = []
label_cache = []
score_cache = []
ups_cache = []
downs_cache = []
comment_wc_cache = []
parent_wc_cache = []

def get_word_count_of_comment(comment):
    return str(len(comment.split())) # Simple word count by splitting on space.

with open('../data/train-unbalanced.csv') as f:
    while True:
        line = f.readline()
        if not line:
            print("Reached end.") # Just to keep track of where we are.
            with open('../data/samples/subreddits_in_training.txt', 'w+') as note:
                json.dump(subreddits_dict, note)
            with open('../data/samples/sample_train_unbalanced.csv', 'a') as note:
                for selected_line in line_cache:
                    note.write(selected_line)
            with open('../data/samples/train_unbalanced_statistics.csv', 'a') as note:
                for i in range(len(score_cache)):
                    line_to_write = label_cache[i] + ',' + score_cache[i] + ',' + ups_cache[i] + ',' + \
                                    downs_cache[i] + ',' + comment_wc_cache[i] + ',' + parent_wc_cache[i] + '\n'
                    note.write(line_to_write)
            print("no more lines")
            break

        split_line = line.split('\t')
        # Firstly, getting our label
        label = split_line[0]
        label_cache.append(label)
        # Then, the subreddit
        subreddit = split_line[3]
        if subreddit not in subreddits_dict:
            subreddits_dict[subreddit] = 0
        subreddits_dict[subreddit] = subreddits_dict[subreddit] + 1
        # Then, the score, ups, and downs
        score = split_line[4]
        ups = split_line[5]
        downs = split_line[6]
        score_cache.append(score)
        ups_cache.append(ups)
        downs_cache.append(downs)

        # Then, the Parent and Child word count
        parent_comment = split_line[9]
        original_comment = split_line[1]
        parent_comment_wc = get_word_count_of_comment(parent_comment)
        original_comment_wc = get_word_count_of_comment(original_comment)
        parent_wc_cache.append(parent_comment_wc)
        comment_wc_cache.append(original_comment_wc)

        # Then, get our sample
        rand_draw = np.random.uniform()
        if rand_draw <= CUTOFF:
            line_cache.append(line)

        if counter % 1000000 == 0:
            print("Reached " + str(counter) + "... Writing results to disk") # Just to keep track of where we are.
            with open('../data/samples/subreddits_in_training.txt', 'w+') as note:
                json.dump(subreddits_dict, note)
            with open('../data/samples/sample_train_unbalanced.csv', 'a') as note:
                for selected_line in line_cache:
                    note.write(selected_line)
            with open('../data/samples/train_unbalanced_statistics.csv', 'a') as note:
                for i in range(len(score_cache)):
                    line_to_write = label_cache[i] + ',' + score_cache[i] + ',' + ups_cache[i] + ',' + \
                                    downs_cache[i] + ',' + comment_wc_cache[i] + ',' + parent_wc_cache[i] + '\n'
                    note.write(line_to_write)

            line_cache = []
            label_cache = []
            score_cache = []
            ups_cache = []
            downs_cache = []
            comment_wc_cache = []
            parent_wc_cache = []

        counter = counter + 1
