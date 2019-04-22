from nltk.corpus import wordnet

sentence = 'hello world my name is hello world'

# im very skeptical of this concerning runtime but whatever
memo = {}
def get_synonyms_and_synset_of_sentence(s):
    synonym_count_list = []
    synset_count_list = []
    for word in s.split(' '):
        if word in memo:
            synonym_count_list.append(memo[word][0])
            synset_count_list.append(memo[word][1])
            continue
        syns = wordnet.synsets(word)
        synset_count_list.append(len(syns))
        synonym_set = set()
        for syn in syns:
            for w in syn.lemmas():
                synonym_set.add(w.name())
        synonym_count_list.append(len(synonym_set))
        if word not in memo:
            memo[word] = (len(synonym_set), len(syns))
    
    max_synset = max(synset_count_list)
    max_synonyms = max(synonym_count_list)
    avg_synset = (sum(synset_count_list) * 1.0)/len(synset_count_list)
    avg_synonyms = (sum(synonym_count_list) * 1.0)/len(synonym_count_list)

    res = (max_synset, max_synonyms, avg_synset, avg_synonyms)
    return res




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

        parent_synset = get_synonyms_and_synset_of_sentence(parent_comment)
        original_sysnset = get_synonyms_and_synset_of_sentence(original_comment)
        feature_arr.append(original_sysnset + parent_synset)
        counter = counter + 1
        if counter % 10000 == 0:
            print(counter)

print('done with iteration. writing results...')

schema = "max_synset_original,max_synonyms_original,avg_synset_original,avg_synonyms_original,max_synset_parent,max_synonyms_parent,avg_synset_parent,avg_synonyms_parent"
with open('../data/features/synset_train.csv', 'w+') as note:
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

        parent_synset = get_synonyms_and_synset_of_sentence(parent_comment)
        original_sysnset = get_synonyms_and_synset_of_sentence(original_comment)
        feature_arr.append(original_sysnset + parent_synset)
        counter = counter + 1
        if counter % 10000 == 0:
            print(counter)

print('done with iteration. writing results...')

schema = "max_synset_original,max_synonyms_original,avg_synset_original,avg_synonyms_original,max_synset_parent,max_synonyms_parent,avg_synset_parent,avg_synonyms_parent"
with open('../data/features/synset_test.csv', 'w+') as note:
    note.write(schema + '\n')
    for i in range(1, len(feature_arr)):
        line = feature_arr[i]
        note.write(str(line)[1:-1] + '\n')


print('done')
