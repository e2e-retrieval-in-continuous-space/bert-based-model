from enum import Enum, unique
from models.simple_embedding_model import SimpleEmbeddingModel
from models.bert_feature_extractor_model import BERTAsFeatureExtractorEncoder, BERTVersion


@unique
class ModelType(Enum):
    BERT_AVG_LAYER_TOKEN_FEATURE_EXTRACTOR = "bert_avg_layer_token_feature_extractor"
    SIMPLE_EMBEDDING_MODEL = "simple_embedding_model"


def get_model(model_type: ModelType, train_config):
    if model_type is ModelType.BERT_AVG_LAYER_TOKEN_FEATURE_EXTRACTOR:
        return BERTAsFeatureExtractorEncoder(
            BERTVersion.BASE_UNCASED,
            hidden_size=train_config["embedding_dim"]
        )

    elif model_type is ModelType.SIMPLE_EMBEDDING_MODEL:
        return SimpleEmbeddingModel(train_config["embedding_dim"])

    else:
        raise Exception("Invalid model type")