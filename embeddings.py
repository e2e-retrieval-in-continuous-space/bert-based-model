import torch
import enum
from typing import List, Callable
from transformers import AutoTokenizer, AutoModel, AutoConfig


class BERTVersion(enum.Enum):
    """
    Enum representing supported BERT versions
    """
    BERT_BASE_UNCASED = 'bert-base-uncased'


class BERTWrapper(object):
    """
    Convenience wrapper. Loads all BERT resources and exposes a `run` method that runs BERT on the input.

    :param model_name: The name of specific model to use (Use `Model` enum)
    """

    def __init__(self, model_name: BERTVersion):
        self.model_name = model_name.value
        self.config = AutoConfig.from_pretrained(self.model_name, output_hidden_states=True, return_dict=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, config=self.config)
        self.model = AutoModel.from_pretrained(self.model_name, config=self.config)

    def run_model(self, sentences: List[str]) -> torch.Tensor:
        """
        Runs BERT on a list of sentences and returns the output

        :param sentences: A list of sentences create embeddings for.
        :return: Tensor of shape [num_layers, num_sentences, num_tokens, num_bert_features].
        """
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        return torch.stack(outputs.hidden_states)


def average_layers_and_tokens(tensor: torch.Tensor) -> torch.Tensor:
    return average_axis(0, average_axis(len(tensor.shape) - 2, tensor))


def average_axis(axis: int, tensor: torch.Tensor) -> torch.Tensor:
    return tensor.sum(axis) / tensor.shape[axis]


def embeddings_factory(bert: BERTWrapper, reducer: Callable[[torch.Tensor], torch.Tensor]) -> Callable[
    [List[str]], torch.Tensor
]:
    def create_embeddings(sentences: List[str]) -> torch.Tensor:
        """
        This function runs BERT on `sentences` and reduces the output from shape (a, b, c, d) to only (a, b) which is
        a matrix of embeddings with rows as vectors.
        """
        return reducer(bert.run_model(sentences))

    return create_embeddings
