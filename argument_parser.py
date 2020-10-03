import argparse
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
	args = argparse.Namespace(
			seed=42,
			data_path='data/yelp.csv',
			train_percent=0.7,
			valid_percent=0.2,
			test_percent=0.1,

			# Model parameters
			model_state_file='model.pth',
			save_dir='models/',

			# Training HyperParameters
			batch_size=128,
			early_stopping_criteria=5,
			learning_rate=0.001,
			num_epochs=100,
			device="cpu"
		)

