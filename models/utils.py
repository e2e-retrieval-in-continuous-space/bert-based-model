import uuid
import pickle
from itertools import chain
from pathlib import Path
from data_utils import flatmap
from loggers import getLogger

CACHE_PATH = Path(__file__).parents[0] / "embeddings-cache"
CACHE_PATH.mkdir(parents=True, exist_ok=True)

logger = getLogger(__name__)

class EmbeddingsCache(dict):

    def __init__(self, name, *args, **kwargs):
        super(EmbeddingsCache, self).__init__(*args, **kwargs)
        self.cache_path = CACHE_PATH / name
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.unsaved_keys = []
        logger.info("Loading cached embeddings from %s", self.cache_path)
        try:
            for i, path in enumerate(self.cache_path.glob('*.pickle')):
                with path.open('rb') as cachefile:
                    self.update(pickle.load(cachefile))
            logger.info("%d cached embeddings loaded from %d files", len(self), i + 1)
        except Exception as e:
            logger.warn("Cached BERT embeddings from %s cannot be loaded (%s)", self.cache_path, e)

    def get_many(self, documents):
        hits = []
        misses = []
        for doc in documents:
            if doc in self:
                hits.append(self[doc])
            else:
                hits.append(None)
                misses.append(doc)
        return hits, misses

    def __setitem__(self, key, value):
        if key not in self:
            self.unsaved_keys.append(key)
        super(EmbeddingsCache, self).__setitem__(key, value)
        if len(self.unsaved_keys) > 10000:
            self.save()

    def save(self):
        nb_items = len(self.unsaved_keys)
        logger.debug("Saving %d embeddings", nb_items)
        new_path = self.cache_path / "{0}.pickle".format(uuid.uuid4())
        with new_path.open('wb') as cachefile:
            tmp_dict = {k: self[k] for k in self.unsaved_keys}
            pickle.dump(tmp_dict, cachefile)
            self.unsaved_keys = []
        logger.debug("Saved %d embeddings", nb_items)


class BagOfWordsTokenizer:

    UNK_TOKEN = '<UNK>'
    PAD_TOKEN = '<PAD>'

    def __init__(self, train_data, test_data, qid2text):
        self.train_data = train_data
        self.test_data = test_data
        self.qid2text = qid2text
        self.vocab = self.build_vocabulary(self.train_data, self.test_data, self.qid2text)
        self.vocab_size = len(self.vocab)
        self.size = 100
        self.pad_token_id = self.vocab[self.PAD_TOKEN]
        self.unk_token_id = self.vocab[self.UNK_TOKEN]

    def __call__(self, sentences):
        return self.tokenize_many(sentences)

    def tokenize_many(self, tokenized):
        tokenized = [self.tokenize_sentence(sentence) for sentence in tokenized]
        padded_sentences = [s + [self.pad_token_id] + [self.unk_token_id] * (self.size + 1 - len(s)) for s in tokenized]
        return LongTensor(padded_sentences)

    def tokenize_sentence(self, sentence):
        return [self.vocab.get(s, self.unk_token_id) for s in self.featurize(sentence)]

    def featurize(self, text):
        return text.lower().split(' ')

    def build_vocabulary(self, train_data, test_data, qid2text):
        vocab = set()

        texts = chain(
            [q.text for q in flatmap(chain(train_data, test_data))],
            qid2text.values()
        )
        for text in texts:
            tokens = self.featurize(text)
            for t in tokens:
                vocab.add(t)

        vocab = [self.UNK_TOKEN, self.PAD_TOKEN] + list(vocab)

        return dict(zip(vocab, range(len(vocab))))
