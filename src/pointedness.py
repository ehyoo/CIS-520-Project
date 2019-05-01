import string

punct_set = set(string.punctuation)

def get_pointed_for_word(w):
    # We only check for number of punctuation marks and full-caps words.
    # Each punctuation mark (or string of punctuation marks) is a +1, and 
    # if the word is capitalized, that's a +1 as well.
    # From Reyes 2013
    
    num_pointed = 0
    last_char_encountered = ''
    does_contain_non_punct = False
    for c in w:
        if c in punct_set:
            if c != last_char_encountered:
                num_pointed = num_pointed + 1
        else:
            does_contain_non_punct = True
        last_char_encountered = c

    
    if (does_contain_non_punct and w.upper() == w):
        num_pointed = num_pointed + 1
    return num_pointed
    

def compute_number_of_pointed_tokens(s):
    # Be sure to feed in the unfiltered version of the text.
    s_arr = s.split()
    num_pointed = 0
    for word in s_arr:
        num_pointed = num_pointed + get_pointed_for_word(word)
    return num_pointed

feature_arr = []
counter = 0
# note we feed in the original data here
with open('../data/pol-train.csv') as f:
    while True:
        line = f.readline()
        if not line:
            break
        split_line = line.split('\t')
        # Then, the Parent and Child word count
        parent_comment = split_line[9]
        original_comment = split_line[1]

        parent_tokens = compute_number_of_pointed_tokens(parent_comment)
        original_tokens = compute_number_of_pointed_tokens(original_comment)
        feature_arr.append((original_tokens, parent_tokens))
        counter = counter + 1
        if counter % 10000 == 0:
            print(counter)

print('done with iteration. writing results...')
schema = "original_pointedness,parent_pointedness"
with open('../data/features/pointedness_train.csv', 'w+') as note:
    note.write(schema + '\n')
    for i in range(len(feature_arr)):
        line = feature_arr[i]
        note.write(str(line)[1:-1] + '\n')

print("Starting test")
feature_arr = []
counter = 0
# note we feed in the original data here
with open('../data/pol-test.csv') as f:
    while True:
        line = f.readline()
        if not line:
            break
        split_line = line.split('\t')
        # Then, the Parent and Child word count
        parent_comment = split_line[9]
        original_comment = split_line[1]

        parent_tokens = compute_number_of_pointed_tokens(parent_comment)
        original_tokens = compute_number_of_pointed_tokens(original_comment)
        feature_arr.append((original_tokens, parent_tokens))
        counter = counter + 1
        if counter % 10000 == 0:
            print(counter)

print('done with iteration. writing results...')
schema = "original_pointedness,parent_pointedness"
with open('../data/features/pointedness_test.csv', 'w+') as note:
    note.write(schema + '\n')
    for i in range(len(feature_arr)):
        line = feature_arr[i]
        note.write(str(line)[1:-1] + '\n')

print('done')

