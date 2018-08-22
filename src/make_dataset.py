import numpy as np
import pandas as pd
from pathlib import Path
import utils

# Read data
train_path = Path.cwd().joinpath('raw-data/semeval-2016/train.xml')
test_path = Path.cwd().joinpath('raw-data/semeval-2016/test.xml')

# Convert data to dataframe
train_data = utils.read_data(train_path)
test_data = utils.read_data(test_path)

# Save data as csv
save_train = Path.cwd().joinpath('data/semeval-2016/train.csv')
train_data.to_csv(save_train, index=False)

save_test = Path.cwd().joinpath('data/semeval-2016/test.csv')
test_data.to_csv(save_test, index=False

