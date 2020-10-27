import argparse
from model_factory import ModelFactory
from torch import optim
from train_utils import fit
from quora_dataset import QuoraDataset
from loggers import getLogger
import os
import sys

logger = getLogger(__name__)

"""
# For usage, run:
python train.py --help
"""

default_train_config = {
    "learning_rate": 1e-3,
    "embedding_dim": 300,
    "top_k": 10,
    "batch_size": 100,
    "epoch_num": 5,
}

parser = argparse.ArgumentParser(description='Training a model with Quora dataset')

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
dataset = QuoraDataset(limit=args.limit)
logger.info("Quora dataset loaded")

train_data = dataset.get_train_data()
test_data = dataset.get_test_data()
candidates = dataset.get_candidates()

import torch
import torch.cuda
cuda = torch.device('cuda')
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.autograd.set_detect_anomaly(True)
model = ModelFactory.get_model(args.model_type, vars(args))

# @TODO: Change to a different optimizer
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

logger.info("Command-line args %s", args)
logger.info("Running fit() for model %s", model.__class__.__name__)

# Start fitting the model
fit(
    epochs=args.epoch_num,
    model=model,
    opt=optimizer,
    train_data=train_data,
    test_data=test_data,
    dataset=dataset,
    candidates=candidates,
    top_k=args.top_k,	
    batch_size=args.batch_size	,
    save_model_dir=save_model_dir
)

