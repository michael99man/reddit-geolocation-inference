import featurization.mysql as mysql
from collections import defaultdict
import json
import time
import pickle
import multiprocessing
from gensim.models import Word2Vec
import logging  # Setting up the loggings to monitor gensim
from gensim.models.callbacks import CallbackAny2Vec
from gensim.test.utils import get_tmpfile
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)


class EpochSaver(CallbackAny2Vec):
    '''Callback to save model after each epoch.'''
    def __init__(self, path_prefix):
        self.path_prefix = path_prefix
        self.epoch = 0

    def on_epoch_end(self, model):
        filename = '{}_epoch{}.model'.format(self.path_prefix, self.epoch)
        output_path = get_tmpfile(filename)
        print("Saved %s" % filename)
        model.save(output_path)
        self.epoch += 1


def main():
    rows = mysql.fetch_all_cleaned(limit=None)
    print("Fetched %d rows" % len(rows))

    documents = extract_documents(rows)
    word2vec(documents)


def word2vec(documents):
    saver = EpochSaver("my_w2v")

    cores = multiprocessing.cpu_count()
    model = Word2Vec(min_count=100,
                         # skip gram
                         sg=1,
                         # window size
                         window=3,
                         # number of parameters
                         size=300,
                         # downsample frequent words
                         sample=5e-5,
                         negative=20,
                         workers=cores - 1,
                         callbacks=[saver]
                     )

    t = time.time()
    model.build_vocab(documents, progress_per=10000)

    print('Time to build vocab: {} mins'.format(round((time.time() - t) / 60, 2)))
    model.save("vocab.model")

    t = time.time()
    model.train(documents, total_examples=model.corpus_count, epochs=30, report_delay=1)
    print('Time to train the model: {} mins'.format(round((time.time() - t) / 60, 2)))
    model.save("full.model")


# extracts documents from rows, converting the stringified list into a list
def extract_documents(rows):
    print("Extracing documents...")
    documents = []
    for i in range(len(rows)):
        # convert list string into list
        d = rows[i]['document'].replace("\'", "\"")
        tokens = json.loads(d)

        documents.append(tokens)
        if (i % 10000 == 0):
            print("%d/%d: %d tokens" % (i, len(rows), len(tokens)))
    return documents

# compute dictionary of all words
def compute_dict(documents):
    dictionary = defaultdict(int)
    for i in range(documents):
        for t in documents[i]:
            dictionary[t] += 1

    print(len(dictionary.keys()))
    pickle.dump(dictionary, open('dict.p', 'wb'))

if __name__ == "__main__":
    main()
