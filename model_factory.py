from models.simple_embedding_model import SimpleEmbeddingModel
from models.bert_feature_extractor_model import BERTAsFeatureExtractorEncoder, BERTVersion, reducer_all_layers, \
    reducer_last_layer, reducer_2nd_last_layer, reducer_last_4_layers, reducer_try_vertical_tokens


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
    def bert_vertical_tokens(train_config):
        return ModelFactory._bert_base(train_config, reducer_try_vertical_tokens)

    @staticmethod
    def bert_base_all_layers(train_config):
        return ModelFactory._bert_base(train_config, reducer_all_layers)

    @staticmethod
    def bert_base_last_layers(train_config):
        return ModelFactory._bert_base(train_config, reducer_last_layer)

    @staticmethod
    def bert_base_2nd_last_layer(train_config):
        return ModelFactory._bert_base(train_config, reducer_2nd_last_layer)

    @staticmethod
    def bert_base_last_4_layers(train_config):
        return ModelFactory._bert_base(train_config, reducer_last_4_layers)

    @staticmethod
    def bert_large_all_layers(train_config):
        return ModelFactory._bert_large(train_config, reducer_all_layers)

    @staticmethod
    def bert_large_last_layers(train_config):
        return ModelFactory._bert_large(train_config, reducer_last_layer)

    @staticmethod
    def bert_large_2nd_last_layer(train_config):
        return ModelFactory._bert_large(train_config, reducer_2nd_last_layer)

    @staticmethod
    def bert_large_last_4_layers(train_config):
        return ModelFactory._bert_large(train_config, reducer_last_4_layers)

    @staticmethod
    def _bert_base(train_config, reducer):
        return BERTAsFeatureExtractorEncoder(
            BERTVersion.BASE_UNCASED,
            hidden_size=train_config["embedding_dim"],
            bert_reducer=reducer
        )

    @staticmethod
    def _bert_large(train_config, reducer):
        return BERTAsFeatureExtractorEncoder(
            BERTVersion.LARGE_UNCASED,
            hidden_size=train_config["embedding_dim"],
            bert_reducer=reducer
        )

    @staticmethod
    def simple_embedding_model(train_config):
        return SimpleEmbeddingModel(train_config["embedding_dim"])
