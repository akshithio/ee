import random

txt_corpus = open("data/raw/corpus.txt", "r")

txt_ds_1 = open("data/processing/data/50m.txt", "w+")
txt_ds_2 = open("data/processing/data/100m.txt", "w+")
txt_ds_3 = open("data/processing/data/150m.txt", "w+")

txt_indices_ds_1 = open("data/processing/indices/50m.txt", "w+")
txt_indices_ds_2 = open("data/processing/indices/100m.txt", "w+")
txt_indices_ds_3 = open("data/processing/indices/150m.txt", "w+")

corpus = txt_corpus.readlines()
corpus_indices = len(corpus) # one index corresponds to one line

# word_count variables

word_count_corpus = 0
word_count_ds_1 = 0
word_count_ds_2 = 0
word_count_ds_3 = 0

# check number of lines and words on corpus 

print("no. of lines in the entire corpus: ", corpus_indices) #7,871,825

for line in corpus:
    words = line.split()
    word_count_corpus += len(words)

print("no. of words in the entire corpus: ", word_count_corpus) #151,523,090

# prepare all indices_ds_num

indices = set([i for i in range (0,corpus_indices)]) #len(indices) = 7,871,825
indices_ds_3 = indices.copy() #ds3 refers to the 150m words dataset which is the corpus itself, dt: set


indices_ds_1  = random.sample(indices, len(indices) // 3) # dt: list

for num in indices_ds_1:
    indices.remove(num)


print("number of indices after removing those from ds_1: ", len(indices)) # should be 5,247,884
 
temp_indices_ds_2 = random.sample(indices, len(indices) // 2)
indices_ds_2 = indices_ds_1 + temp_indices_ds_2 # dt: list

# verify to ensure there's no duplicate-indexing

if len(set(indices_ds_1)) == len(indices_ds_1) and len(set(indices_ds_2)) == len(indices_ds_2):
    print("no duplicate indexing detected")

#check indice length

print("number of lines (derived from indices) in dataset 1: ", len(indices_ds_1)) #2,623,941
print("number of lines (derived from indices) in dataset 2: ", len(indices_ds_2)) #6,559,853
print("number of lines (derived from indices) in dataset 3: ", len(indices_ds_3)) #5,247,884

# check if indices_ds_2's first x number of elements are exactly as mentioned in indices_ds_1 (x = 2,623,941)

error_count = 0
error_indices = []

for i in range (0, 2623941):
    if indices_ds_1[i] != indices_ds_2[i]:
        error_count += 1
        error_indices.append(i)

if error_count > 0:
    print(error_count)
    print(error_indices)
else:
    print("ds_1 is a confirmed subset of ds_2")

#write indices to indice txt files 

txt_indices_ds_1.write(str(indices_ds_1))
txt_indices_ds_2.write(str(indices_ds_2))
txt_indices_ds_3.write(str(indices_ds_3))

#write to all 3 datasets

for indice in indices_ds_1:
    txt_ds_1.write(str(corpus[indice]))

for indice in indices_ds_2:
    txt_ds_2.write(str(corpus[indice]))

for indice in indices_ds_3:
    txt_ds_3.write(str(corpus[indice]))

# read dataset streams to check word_count (erroring)

txt_ds_1.seek(0)
ds_1 = txt_ds_1.readlines()
txt_ds_2.seek(0)
ds_2 = txt_ds_2.readlines()
txt_ds_3.seek(0)
ds_3 = txt_ds_3.readlines()

# check word_count

word_count_ds_1 = sum(len(line.split()) for line in ds_1)
word_count_ds_2 = sum(len(line.split()) for line in ds_2)
word_count_ds_3 = sum(len(line.split()) for line in ds_3)

# word_counts for all datasets

print("word count for dataset 1: ", word_count_ds_1)
print("word count for dataset 2: ", word_count_ds_2)
print("word count for dataset 3: ", word_count_ds_3)
