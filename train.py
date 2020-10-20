from embeddings.bert import BERTWrapper, BERTVersion, average_layers_and_tokens, bert_embeddings_factory
from embeddings.baselines import compute_random_embeddings, compute_tfidf_features
from train_utils import Encoder, DualEncoder

bert = BERTWrapper(BERTVersion.BERT_BASE_UNCASED)
embeddings_dim = bert.config.hidden_size
compute_bert_embeddings = bert_embeddings_factory(bert, average_layers_and_tokens)

sentences = [
    "We plan to apply the feature-based approach with BERT (Devlin et al. 2018) ",
    "By extracting the final layer’s activations without updating.",
    "We would average.",
    "This sequence of hidden states for the whole input sequence to summarize the semantic content of the input."
]

encoder = Encoder(compute_bert_embeddings, embeddings_dim * 2)
print(encoder(sentences))

dual = DualEncoder(encoder)
print(dual(sentences, sentences))

print(compute_tfidf_features(sentences))
print(compute_bert_embeddings(sentences))
print(compute_random_embeddings(sentences))
