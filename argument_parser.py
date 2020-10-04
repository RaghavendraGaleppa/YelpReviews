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

	parser.add_argument('-msf', '--model-state-file',
			help="The path to the model state file",
			default="model.pth",
			)

	parser.add_argument('-msd', '--model-save-dir',
			help="The directory in which new models will be saved",
			default="model/",
			)

	parser.add_argument('-bs', '--batch-size',
			help="The batch to train on",
			type=int,
			default=128,
			)

	parser.add_argument('-esc', '--early-stopping-criteria',
			help="When to early stop the training",
			type=int,
			default=5,
			)

	parser.add_argument('-lr', '--learning-rate',
			help="The learning rate of the optimizer",
			type=float,
			default=0.001,
			)

	parser.add_argument('-ne', '--num-epochs',
			help="The number of epochs to train on",
			type=int,
			default=100,
			)

	parser.add_argument('-d', '--device',
			help="The device. It should be either cuda or cpu",
			default=None,
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
			model_save_dir='models/',

			# Training HyperParameters
			batch_size=128,
			early_stopping_criteria=5,
			learning_rate=0.001,
			num_epochs=100,
			device=None,
		)

