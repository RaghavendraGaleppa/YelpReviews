import argparse
import pandas as pd
import numpy as np
import collections

# Create an argument parser
if __name__ == "__main__":
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
else:
	class ArgumentParser():
		def __init__(self,
				seed=42,
				data_path='data/yelp.csv',
				train_percent=0.7,
				valid_percent=0.2,
				test_percent=0.1,
				):
			self.seed = seed
			self.data_path = data_path
			self.train_percent = train_percent
			self.test_percent = test_percent
			self.valid_percent = valid_percent

		def __str__(self):
			return f"seed={self.seed}, train_percent={self.train_percent}, " +\
				f"test_percent={self.test_percent}, valid_percent={self.valid_percent}"

	args = ArgumentParser()



