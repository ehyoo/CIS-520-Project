
lines = []
counter = 0
with open('../data/samples/tts_sample_train.csv') as f:
    for i in range(100000):
        line = f.readline()
        if not line:
            print("Reached end.") # Just to keep track of where we are.
            break
        
        lines.append(line)
        counter = counter + 1
        if counter % 10000 == 0:
            print(counter)

with open('../data/samples/tts_subset.csv', 'w+') as note:
    for line in lines:
        note.write(line + '\n')