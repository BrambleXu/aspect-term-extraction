import pandas as pd
from pathlib import Path
import anago
from anago.utils import load_glove
import utils


train_path = Path.cwd().joinpath('data/semeval-2016/train.csv')
test_path = Path.cwd().joinpath('data/semeval-2016/test.csv')

# Read data
data_train = pd.read_csv(train_path)
data_test = pd.read_csv(test_path)

x_train, y_train = utils.df2data(data_train)
x_test, y_test = utils.df2data(data_test)

# Load glove embedding
EMBEDDING_PATH = '../embedding_weights/glove.840B.300d.txt'
embeddings = load_glove(EMBEDDING_PATH)

# Use pre-trained word embeddings to train
model = anago.Sequence(embeddings=embeddings, word_embedding_dim=300)
model.fit(x_train, y_train, x_test, y_test, epochs=10)

