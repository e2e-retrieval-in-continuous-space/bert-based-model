import uuid
import pickle
from pathlib import Path
from loggers import getLogger

CACHE_PATH = Path(__file__).parents[0] / "embeddings-cache"
CACHE_PATH.mkdir(parents=True, exist_ok=True)

logger = getLogger(__name__)

class EmbeddingsCache(dict):

    def __init__(self, name, *args, **kwargs):
        super(EmbeddingsCache, self).__init__(*args, **kwargs)
        self.cache_path = CACHE_PATH / name
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

