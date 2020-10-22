from models import BERTAsFeatureExtractorEncoder, BERTVersion
from torch import optim
from train_utils import fit


train_data = [
    ("What is the greatest mystery in the universe?", "I would like to know about universe's mysteries."),
    ("What is the greatest mystery in the universe?", "What do you think is the universe's greatest mystery?"),
]

test_data = [
    ("What is square root of 9?", "I would like to know the square root of 9."),
    ("What is square root of 9?", "Anyone knows square root of 9?"),
]

learning_rate = 1e-1
model = BERTAsFeatureExtractorEncoder(BERTVersion.BASE_UNCASED)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
fit(
    epochs=5,
    model=model,
    opt=optimizer,
    train_data=train_data,
    test_data=test_data,
    candidates=("square", "root")
)
