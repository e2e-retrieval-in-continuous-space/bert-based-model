from models import BERTAsFeatureExtractorEncoder, BERTVersion
from torch import optim
from train_utils import fit
from quora_dataset import QuoraDataset
from loggers import getLogger

logger = getLogger(__name__)

logger.info("Loading Quora dataset...")
dataset = QuoraDataset(limit=100)
logger.info("Quora dataset loaded")

train_data = dataset.get_train_data()
test_data = dataset.get_test_data()
candidates = dataset.get_candidates()[:10]

learning_rate = 1e-1
model = BERTAsFeatureExtractorEncoder(
    BERTVersion.BASE_UNCASED,
    hidden_size=300
)
# @TODO: Change to a different optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
logger.info("Running fit()")
fit(
    epochs=5,
    model=model,
    opt=optimizer,
    train_data=train_data,
    test_data=test_data,
    dataset=dataset,
    candidates=candidates,
    top_k=2,
    batch_size=10,
)
