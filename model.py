from embeddings.bert import BERTWrapper, BERTVersion, average_layers_and_tokens, bert_embeddings_factory
from embeddings.random import compute_random_embeddings

bert = BERTWrapper(BERTVersion.BERT_BASE_UNCASED)
compute_bert_embeddings = bert_embeddings_factory(bert, average_layers_and_tokens)

sentences = [
    "We plan to apply the feature-based approach with BERT (Devlin et al. 2018) ",
    "By extracting the final layer’s activations without updating.",
    "We would average.",
    "This sequence of hidden states for the whole input sequence to summarize the semantic content of the input."
]

print(compute_bert_embeddings(sentences))
print(compute_random_embeddings(sentences))
