import argparse
from model_factory import ModelFactory
from torch import optim
from train_utils import fit
from data_utils import chunks, flatmap
from quora_dataset import QuoraDataUtil
from loggers import getLogger
import os
import sys

logger = getLogger(__name__)

"""
# For usage, run:
python train.py --help
"""

default_train_config = {
    "learning_rate": 1e-2,
    "embedding_dim": 300,
    "top_k": 100,
    "batch_size": 1000,
    "retrieval_batch_size": 1000,
    "epoch_num": 15
}

parser = argparse.ArgumentParser(description='Training a model with Quora dataset')

parser.add_argument('command',
                    type=str,
                    choices=['precompute-sentence-embeddings', 'precompute-word-embeddings', 'train'],
                    default='train',
                    help='Command to run')

parser.add_argument('--model_type',
                    type=str,
                    choices=ModelFactory.get_available_models(),
                    default=ModelFactory.simple_embedding_model.__name__,
                    help='Encoder model type')

parser.add_argument('--epoch_num',
                    type=int,
                    default=default_train_config['epoch_num'],
                    help='Number of passes of the dataset to train the model')

parser.add_argument('--embedding_dim',
                    type=int,
                    default=default_train_config['embedding_dim'],
                    help='Number of passes of the dataset to train the model')

parser.add_argument('--top_k',
                    type=int,
                    default=default_train_config['top_k'],
                    help='Number of top candidates to compute MAP@K')

parser.add_argument('--batch_size',
                    type=int,
                    default=default_train_config['batch_size'],
                    help='Batch size for training')

parser.add_argument('--retrieval_batch_size',
                    type=int,
                    default=default_train_config['retrieval_batch_size'],
                    help='Batch size for evaluation of test query')

parser.add_argument('--learning_rate',
                    type=float,
                    default=default_train_config['learning_rate'],
                    help='Learning rate for the optimizer')

parser.add_argument('--limit',
                    type=int,
                    help='Limit for the dataset')

parser.add_argument('--save_model_dir',
                    type=str,
                    help='Directory to save model parameters after training')

args = parser.parse_args()

# If the save_model_dir doesn't exist or not writable, fail now
save_model_dir = args.save_model_dir
if save_model_dir and (not os.path.isdir(save_model_dir) or not os.access(save_model_dir, os.W_OK)):
    sys.stderr.write("{} is not a writable directory\n".format(args.save_model_dir))
    sys.exit(1)


logger.info("Loading Quora dataset...")
data_loader = QuoraDataUtil(limit=args.limit)
model = ModelFactory.get_model(args.model_type, vars(args))

if args.command == "precompute-sentence-embeddings":
    logger.info("==== Precomputing sentence embeddings ====")

    logger.info("Processing examples...")
    unique_texts = list(set(flatmap([(e.q1_text, e.q2_text) for e in data_loader.get_examples()])))
    examples_chunks = list(chunks(unique_texts, int(args.batch_size)))
    for i, chunk in enumerate(examples_chunks):
        logger.debug("Chunk %d out of %d", i, len(examples_chunks))
        model.compute_sentence_embeddings(chunk)
        model.cache.save()
elif args.command == "precompute-word-embeddings":
    logger.info("==== Precomputing word embeddings ====")

    logger.info("Processing examples...")
    unique_texts = list(set(flatmap([(e.q1_text, e.q2_text) for e in data_loader.get_examples()])))
    examples_chunks = list(chunks(unique_texts, int(args.batch_size)))
    for i, chunk in enumerate(examples_chunks):
        logger.debug("Chunk %d out of %d", i, len(examples_chunks))
        model.compute_word_embeddings_batch(chunk)
    embeddings = model.tokens_embeddings_to_vocab_embeddings()
else:
    train_data, test_data, retrieval_data, candidate_ids, qid2text = data_loader.construct_retrieval_task()

    # @TODO: Change to a different optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    logger.info("Command-line args %s", args)
    logger.info("train_data count: %d", len(train_data))
    logger.info("test_data count: %d", len(test_data))
    logger.info("retrieval count: %d", len(retrieval_data))
    candidate_per_query = sum([len(result) for query, result in retrieval_data])/len(retrieval_data)
    logger.info("relevant candidates per query: %.2f", candidate_per_query)
    logger.info("candidate_ids count: %d", len(candidate_ids))
    logger.info("qid2text count: %d", len(qid2text))
    logger.info("Running fit() for model %s", model.__class__.__name__)

    logger.info("==== Training the model ====")
    # Start fitting the model
    fit(
        epochs=args.epoch_num,
        model=model,
        opt=optimizer,
        train_data=train_data,
        test_data=test_data,
        retrieval_data=retrieval_data,
        candidate_ids=candidate_ids,
        qid2text=qid2text,
        top_k=args.top_k,
        batch_size=args.batch_size,
        retrieval_batch_size=args.retrieval_batch_size,
        save_model_dir=save_model_dir
    )

