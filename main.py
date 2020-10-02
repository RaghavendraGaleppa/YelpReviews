import argparse
import pandas as pd
import numpy as np
import collections

from tqdm import tqdm

from utils import preprocess_text

# Create an argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seed', 
		help="The random seed that will used to generate all random numbers throughout the program",
		type=int,
		default=42,
		)

parser.add_argument('-dp', '--data-path',
		help="Path to the dataset",
		default="data/yelp.csv",
		)

parser.add_argument('-tp', '--train-percent',
		help="Training split percentage",
		type=float,
		default=0.7,
		)

parser.add_argument('-vp', '--valid-percent',
		help="Validation split percentage",
		type=float,
		default=0.2,
		)

parser.add_argument('-tep', '--test-percent',
		help="Testing split percentage",
		type=float,
		default=0.1,
		)

args = parser.parse_args()
print(args)

# Loading the data
reviews_dataset = pd.read_csv(args.data_path)

# Splitting the dataset into train, valid and test
by_ratings = collections.defaultdict(list)
for row_index, row in reviews_dataset.iterrows():
	by_ratings[row.stars].append(row.to_dict())


final_list = []
np.random.seed(args.seed)

for stars, item_list in by_ratings.items():
	np.random.shuffle(item_list)	

	n_total = len(item_list)
	n_train = int(args.train_percent * n_total)
	n_valid = int(args.valid_percent * n_total)
	n_test = int(args.test_percent * n_total)
	
	for item in item_list[:n_train]:
		item['split'] = 'train'

	for item in item_list[n_train:n_train+n_valid]:
		item['split'] = 'valid'

	for item in item_list[n_train+n_valid:n_total]:
		item['split'] = 'test'

	final_list.extend(item_list)

final_reviews = pd.DataFrame(final_list)
final_reviews.review = final_reviews.review.apply(preprocess_text)
