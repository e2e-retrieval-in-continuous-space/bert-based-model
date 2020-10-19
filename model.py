from embeddings import BERTWrapper, BERTVersion, average_layers_and_tokens, embeddings_factory

bert = BERTWrapper(BERTVersion.BERT_BASE_UNCASED)
create_embeddings = embeddings_factory(bert, average_layers_and_tokens)

embeddings = create_embeddings([
    "We plan to apply the feature-based approach with BERT (Devlin et al. 2018) ",
    "By extracting the final layer’s activations without updating.",
    "We would average.",
    "This sequence of hidden states for the whole input sequence to summarize the semantic content of the input."
])

print(embeddings.shape)
