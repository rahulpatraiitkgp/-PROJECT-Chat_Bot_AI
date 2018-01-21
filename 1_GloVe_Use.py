from scipy import spatial
from sys import stdin
import re
import time
import numpy as np
import os

working_dir = r'.'

## Read and load vectors for words ##
vectors = {}
def load_vectors(vectors_file):
    print('Loading word vectors . . .')
    start = time.time()
    
    count = 0
    with open(vectors_file, 'r',encoding = "UTF-8") as f:
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = np.asarray([float(x) for x in vals[1:]], dtype=np.float64)
            count += 1

    end = time.time()
    print('Loaded '+ str(count) +' vectors in ' + str(end - start))
    
## Get cosine similarity using scipy's distance function ##
def Words_CosineSimilarity(w1, w2):
    if w1 not in vectors:
        print(w1 + ' not in vocab')
        return 0
    elif w2 not in vectors:
        print(w2 + ' not in vocab')
        return 0

    w1_vector = vectors[w1]
    w2_vector = vectors[w2]

    return 1 - spatial.distance.cosine(w1_vector, w2_vector)

## sentence similarity based on averaging of the vectors ##
def GetSentenceVector(s1):
    s1 = s1.lower()
    tokens = re.findall(r"\w+|[^\w\s]", s1, re.UNICODE)
    
    token_vectors = np.array([vectors[w] for w in tokens if w in vectors])
    if token_vectors.size == 0:
        return None
    
    average_vector = np.mean(token_vectors, axis=0)

    
    return average_vector

def Sentences_CosineSimilarity(s1, s2):
    s1_vector = GetSentenceVector(s1)
    s2_vector = GetSentenceVector(s2)

    if (s1_vector is not None) and (s2_vector is not None):
        return 1 - spatial.distance.cosine(s1_vector, s2_vector)
    
    return 0

if __name__ == '__main__':
    load_vectors(os.path.join(working_dir, 'glove.6B', 'glove.6B.50d.txt'))
    
    ## Play with sentences ##
    while True:
        print('Enter sentence 1: ', end='', flush=True)
        sentence_1 = stdin.readline().strip('\n').strip()

        if sentence_1 == '':
            break

        print('Enter sentence 2: ', end='', flush=True)
        sentence_2 = stdin.readline().strip('\n').strip()

        print('Similarity Score: ' + str(Sentences_CosineSimilarity(sentence_1, sentence_2)) + '\n')