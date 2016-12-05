# SOF


# ------------------------------- #
def load_data_shakespeare(vocabulary_size=6000):
    
    corpus = process_data_shakespeare()
    
    dictionary, reverse_dictionary = create_vocabulary_shakespeare(vocabulary_size=vocabulary_size)
    
    data = create_dataset_shakespeare(corpus=corpus, dictionary=dictionary, vocabulary_size=vocabulary_size)
    ydata = create_shifted_dataset(data=data)

    return data, ydata, dictionary, reverse_dictionary
    # ------------------------------- #


# ------------------------------- #
def process_data_shakespeare(RAW_FILENAME='data/shakespeare-full.txt', CORPUS_FILENAME='data/shakespeare-corpus.json'):

    import os
    import json

    if os.path.isfile(CORPUS_FILENAME):
        # Load processed corpus.
        print 'Loading processed corpus...'
        f = open(CORPUS_FILENAME, 'r+')
        corpus = json.loads(f.read())
        f.close()
    else:
        # Open raw data.
        print 'Loading raw data...'
        f = open(RAW_FILENAME, 'r+')
        data_raw = f.read()
        f.close()
        # Process raw data.
        print 'Preparing corpus...'
        corpus = []
        for sentence in data_raw.lower().split('\n'):
            if sentence != '' and len(sentence) > 0:
                ss = str(sentence).translate(string.maketrans("",""), string.punctuation)
                corpus.append(WhitespaceTokenizer().tokenize(ss))
        # Save corpus
        f = open(CORPUS_FILENAME, 'w+')
        f.write(json.dumps(corpus))
        f.close()
        
    print 'The corpus contains {} sentences'.format(len(corpus))

    return corpus
    # ------------------------------- #


# ------------------------------- #
def create_vocabulary_shakespeare(vocabulary_size=6000):
    
    import os
    import json
    import collections

    DICTIONARY_FILENAME = 'data/shakespeare-dictionary-' + str(vocabulary_size) + '.json'

    if os.path.isfile(DICTIONARY_FILENAME):
        # Load processed corpus.
        print 'Loading dictionary for a vocabulary of ' + str(vocabulary_size) + ' from file...'
        f = open(DICTIONARY_FILENAME, 'r+')
        dictionary = json.loads(f.read())
        f.close()
    else:
        print 'Creating dictionary for a vocabulary of ' + str(vocabulary_size) + ' words...'
        all_words = sum(corpus, [])
        print ' --- Found {} words'.format(len(all_words))
        count = [('<UNK>', -1)]
        count.append(('<START>', -1))
        count.append(('<END>', -1))
        count.append(('<PAD>', -1))
        count.extend(collections.Counter(all_words).most_common(vocabulary_size - 1))
        # Create a dictionary from the words.
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        # Save dictionary
        f = open(DICTIONARY_FILENAME, 'w+')
        f.write(json.dumps(dictionary))
        f.close()

    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return dictionary, reverse_dictionary
    # ------------------------------- #


# ------------------------------- #
def create_dataset_shakespeare(corpus, dictionary, vocabulary_size):
    
    import os
    import numpy as np

    DATASET_FILENAME = 'data/shakespeare-dataset-' + str(vocabulary_size) + '.npy'

    if os.path.isfile(DATASET_FILENAME):
        print 'Loading existing datafile...'
        data = np.load(DATASET_FILENAME)
    else:
        print 'Creating new datafile...'
        # Now we need to build the dataset.
        data = []
        for sentence in corpus:
            # sentence start.
            data.append(dictionary['<START>'])
            for word in sentence:
                if word in dictionary:
                    data.append(dictionary[word])
                else:
                    data.append(dictionary['<UNK>'])
            # sentence end.
            data.append(dictionary['<END>'])
        # Save the dataset.
        np.save(DATASET_FILENAME, data)

    return data
    # ------------------------------- #


# ------------------------------- #
def create_shifted_dataset(data):

    import numpy as np

    ydata = np.copy(data)
    ydata[:-1] = data[1:]
    ydata[-1] = data[0]
    
    return ydata
    # ------------------------------- #


# EOF...