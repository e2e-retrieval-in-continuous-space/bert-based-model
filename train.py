import argparse
from model_factory import get_model, ModelType
from torch import optim
from train_utils import fit
from quora_dataset import QuoraDataset
from loggers import getLogger

logger = getLogger(__name__)

logger.info("Loading Quora dataset...")
dataset = QuoraDataset(limit=1000)
logger.info("Quora dataset loaded")

train_data = dataset.get_train_data()
test_data = dataset.get_test_data()
candidates = dataset.get_candidates()

train_config = {
    "learning_rate": 1e-3,
    "embedding_dim": 300,
    "top_k": 10,
    "batch_size": 100,
    "epoch_num": 5,
}

model = get_model(ModelType.SIMPLE_EMBEDDING_MODEL, train_config)

print(model.__class__.__name__)

# @TODO: Change to a different optimizer
optimizer = optim.Adam(model.parameters(), lr=train_config["learning_rate"])
logger.info("Running fit()")
fit(
    epochs=train_config["epoch_num"],
    model=model,
    opt=optimizer,
    train_data=train_data,
    test_data=test_data,
    dataset=dataset,
    candidates=candidates,
    top_k=train_config["top_k"],
    batch_size=train_config["batch_size"],
)
