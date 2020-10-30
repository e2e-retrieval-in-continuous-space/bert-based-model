from models.bow_embedding_model import BagOfWordsEmbeddingModel
from models.glove_model import GloveEmbeddingsModel
from models.simple_embedding_model import SimpleEmbeddingModel
from models.bert_feature_extractor_model import BERTAsFeatureExtractorEncoder, BERTVersion


class ModelFactory:

    @classmethod
    def get_model(cls, model_type, train_config, **kwargs):
        if model_type not in cls.__dict__:
            raise Exception("Invalid model type")
        return getattr(cls, model_type)(train_config, **kwargs)

    @classmethod
    def get_available_models(cls):
        reserved_methods = {"get_model", "get_available_models"}
        factory_methods = list(set(cls.__dict__.keys()) - reserved_methods)
        factory_methods = [m for m in factory_methods if m[0] != "_"]
        return factory_methods

    @staticmethod
    def bert_avg_layer_token_feature_extractor(train_config, **kwargs):
        return BERTAsFeatureExtractorEncoder(
            BERTVersion.BASE_UNCASED,
            hidden_size=train_config["embedding_dim"]
        )

    @staticmethod
    def glove(train_config,
              train_data,
              test_data,
              qid2text,
              **kwargs):
        return GloveEmbeddingsModel(
            train_data,
            test_data,
            qid2text,
            hidden_size=train_config["embedding_dim"]
        )

    @staticmethod
    def simple_embedding_model(train_config, **kwargs):
        return SimpleEmbeddingModel(train_config["embedding_dim"])

    @staticmethod
    def bow_embeddings(train_config,
                       train_data,
                       test_data,
                       qid2text,
                       **kwargs):
        return BagOfWordsEmbeddingModel(
            train_config["embedding_dim"],
            train_data=train_data,
            test_data=test_data,
            qid2text=qid2text,
        )

