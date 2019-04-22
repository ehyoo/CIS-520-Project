def get_frequency_metrics(s):
    word_freq_dict = {}
    s_arr = s.split(' ')
    for word in s_arr:
        if word not in word_freq_dict:
            word_freq_dict[word] = 0
        word_freq_dict[word] = word_freq_dict[word] + 1
    
    # First, get the frequency of the rarest word:
    min_frequency = min(word_freq_dict.values())
    # then get the average frequency of the words in the tweet
    avg_freq = (sum(word_freq_dict.values()) * 1.0)/len(word_freq_dict.values())

    return (min_frequency, avg_freq)

feature_arr = []
counter = 0
with open('../data/samples/pol_train_cleaned.csv') as f:
    while True:
        line = f.readline()
        if not line:
            break
        split_line = line.split('\t')
        # Then, the Parent and Child word count
        parent_comment = split_line[9]
        original_comment = split_line[1]

        parent_synset = get_frequency_metrics(parent_comment)
        original_sysnset = get_frequency_metrics(original_comment)
        feature_arr.append(original_sysnset + parent_synset)
        counter = counter + 1
        if counter % 10000 == 0:
            print(counter)

schema = "min_frequency_original,avg_frequency_original,min_frequency_parent,avg_frequency_parent"
with open('../data/features/frequency_train.csv', 'w+') as note:
    note.write(schema + '\n')
    for i in range(1, len(feature_arr)):
        line = feature_arr[i]
        note.write(str(line)[1:-1] + '\n')

feature_arr = []
counter = 0
with open('../data/samples/pol_test_cleaned.csv') as f:
    while True:
        line = f.readline()
        if not line:
            break
        split_line = line.split('\t')
        # Then, the Parent and Child word count
        parent_comment = split_line[9]
        original_comment = split_line[1]

        parent_synset = get_frequency_metrics(parent_comment)
        original_sysnset = get_frequency_metrics(original_comment)
        feature_arr.append(original_sysnset + parent_synset)
        counter = counter + 1
        if counter % 10000 == 0:
            print(counter)

schema = "min_frequency_original,avg_frequency_original,min_frequency_parent,avg_frequency_parent"
with open('../data/features/frequency_test.csv', 'w+') as note:
    note.write(schema + '\n')
    for i in range(1, len(feature_arr)):
        line = feature_arr[i]
        note.write(str(line)[1:-1] + '\n')
