from scipy import spatial
from sys import stdin
import re
import time
import numpy as np
from random import randint
import os

working_dir = r'.'

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

def Get_QA_DB(file_path_qa):
    print('Loading word vectors . . .')
    start = time.time()

    Q_to_As = {}

    file = open(file_path_qa)
    for line in file:
        val = line.split('\t')
        if len(val) == 2:
            query, answer = val[0], val[1]

            query = query.lower()
            if query not in Q_to_As:
                Q_to_As[query] = []

            Q_to_As[query].append(answer)

    end = time.time()
    print('Built QA of size '+ str(len(Q_to_As)) +' in ' + str(end - start))

    return Q_to_As

def Get_most_similar_query_above_k(Q_to_As, query, threshold):
    max_similarity = -1.0
    most_similar_query = None

    count = 0
    print('Matched with : ', end='', flush=True)
    for q in Q_to_As:
        similarity_score = Sentences_CosineSimilarity(q, query)
        if similarity_score > max_similarity and similarity_score > threshold:
            max_similarity = similarity_score
            most_similar_query = q
        count += 1
        if count % 10000 == 0:
            print(str(count) + ', ', end='',flush=True)

    print('')

    return most_similar_query, max_similarity

def Play(Q_to_As):
    while True:
        print('User: ', end='',flush=True)
        query = stdin.readline().strip('\n').strip()
        
        if query == '':
            break

        most_similar_query, score = Get_most_similar_query_above_k(Q_to_As, query, 0.5)
    
        if most_similar_query != None:
            answer = Q_to_As[most_similar_query][0]
        else:
            answer = None

        if answer != None:
            print('Bot: ' + answer)
            print('// Most similar query was: ' + most_similar_query)
            print('// Most similar query similarity score was: ' + str(score))
            print('')
        else:
            print('Bot: ' + 'No response buddy!')

def get_and_save_results(Q_to_As, file_path_questions, file_path_results):
    with open(file_path_results,'w') as fw:
        with open(file_path_questions,'r') as f:
            for query in f:
                query = query.strip('\n').strip()
                most_similar_query, score = Get_most_similar_query_above_k(Q_to_As, query, 0)
                response = Q_to_As[most_similar_query][0]
                response = response.strip('\n').strip()
                most_similar_query = most_similar_query.strip('\n').strip()

                fw.write(query + '\t' + response + '\t' + str(score) + '\t' + most_similar_query + '\n')

if __name__ == '__main__':
    load_vectors(os.path.join(working_dir, 'glove.6B', 'glove.6B.50d.txt'))
    Q_to_As = Get_QA_DB(os.path.join(working_dir, 'data','QA_pairs.txt'))

    Play(Q_to_As)