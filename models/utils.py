import uuid
import pickle
from pathlib import Path
from loggers import getLogger

CACHE_PATH = Path(__file__).parents[0] / "embeddings-cache"
CACHE_PATH.mkdir(parents=True, exist_ok=True)

logger = getLogger(__name__)

class DictCache(dict):

    def __init__(self, name, *args, **kwargs):
        load = kwargs.pop('load', True)
        self.filename = kwargs.pop('filename', None)
        super(DictCache, self).__init__(*args, **kwargs)
        self.cache_path = CACHE_PATH / name
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.unsaved_keys = []
        if load:
            self.load()

    def load(self, filename=None):
        if filename is None:
            filename = self.filename
        logger.info("Loading cached data from %s", self.cache_path)
        try:
            i = 0
            if filename:
                self.load_file(self.cache_path / "{0}.pickle".format(filename))
            else:
                for i, path in enumerate(self.cache_path.glob('*.pickle')):
                    self.load_file(path)
            logger.info("%d cached records loaded from %d files", len(self), i + 1)
        except Exception as e:
            logger.warn("Cached records from %s cannot be loaded (%s)", self.cache_path, e)

    def load_file(self, file_path):
        with file_path.open('rb') as cachefile:
            self.update(pickle.load(cachefile))

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
        super(DictCache, self).__setitem__(key, value)
        if len(self.unsaved_keys) > 10000:
            self.save()

    def save(self, filename=None):
        if filename is None:
            filename = self.filename or uuid.uuid4()
        nb_items = len(self.unsaved_keys)
        logger.debug("Saving %d records", nb_items)
        new_path = self.cache_path / "{0}.pickle".format(filename)
        with new_path.open('wb') as cachefile:
            tmp_dict = {k: self[k] for k in self.unsaved_keys}
            pickle.dump(tmp_dict, cachefile)
            self.unsaved_keys = []
        logger.debug("Saved %d records", nb_items)


class EmbeddingsCache(DictCache):
    pass
