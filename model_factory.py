from models.bert_words_embeddings_model import BERTWordsEmbeddingsModel
from models.simple_embedding_model import SimpleEmbeddingModel
from models.bert_feature_extractor_model import BERTAsFeatureExtractorEncoder, BERTVersion

class ModelFactory:

    @classmethod
    def get_model(cls, model_type, train_config):
        if model_type not in cls.__dict__:
            raise Exception("Invalid model type")
        return getattr(cls, model_type)(train_config)

    @classmethod
    def get_available_models(cls):
        reserved_methods = {"get_model", "get_available_models"}
        factory_methods = list(set(cls.__dict__.keys()) - reserved_methods)
        factory_methods = [m for m in factory_methods if m[0] != "_"]
        return factory_methods

    @staticmethod
    def bert_avg_layer_token_feature_extractor(train_config):
        return BERTAsFeatureExtractorEncoder(
            BERTVersion.BASE_UNCASED,
            hidden_size=train_config["embedding_dim"]
        )

    @staticmethod
    def bert_words_embeddings_model(train_config):
        return BERTWordsEmbeddingsModel(
            BERTVersion.BASE_UNCASED,
            hidden_size=train_config["embedding_dim"]
        )

    @staticmethod
    def simple_embedding_model(train_config):
        return SimpleEmbeddingModel(train_config["embedding_dim"])
