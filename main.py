import ai
import utils
import os
import pickle
import sys
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Create an argument parser
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("review", nargs="*", 
			help="The text that needs to classified",
			default=None)

	parser.add_argument('-t', '--train',
			help="Train with the data",
			action="store_true",
			)

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

	parser.add_argument('-vf', '--vectorizer-file',
			help="The path to the vectorizer file, that needs to be loaded",
			default="vectorizer.vec",
			)

	parser.add_argument('-msd', '--model-save-dir',
			help="The directory in which new models will be saved",
			default="models/",
			)

	parser.add_argument('-vsd', '--vectorizer-save-dir',
			help="The directory in which vectorizer will be saved",
			default="vectorizer/",
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

	parser.add_argument('-sm', '--save-model',
			help="Save the model at the end of training",
			action="store_true",
			)
	parser.add_argument('-sv', '--save-vectorizer',
			help="Save the vectorizer after creating it",
			action="store_true",
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
			save_model=False,

			# Vectorizer
			vectorizer_file='vectorizer.vec',
			vectorizer_save_dir='vectorizer/',
			save_vectorizer=False,

			# Training HyperParameters
			batch_size=128,
			early_stopping_criteria=5,
			learning_rate=0.001,
			num_epochs=100,
			device=None,
		)

if args.train:
	print(f"Training Mode")
	print(f"="*40)

else:
	print(f"Inference Mode")
	print(f"="*40)
	print("\n")
	args.review = " ".join(args.review)
	print(f"Text: {args.review}")


if args.device is None:
	if torch.cuda.is_available():
		args.device = "cuda"
		print(f"Using GPU")
	else:
		args.device = "cpu"
		print(f"Using CPU")

# Dataset and Vectorizer
dataset = utils.ReviewDataset.create_and_make_dataset(args)
vectorizer = dataset.get_vectorizer()


# Save the vectorizer if flag is given
if args.save_vectorizer:
	if not os.path.isdir(args.vectorizer_save_dir):
		os.makedirs(args.vectorizer_save_dir)
	serializer = vectorizer.to_serializer()
	now = datetime.now()
	file_path = os.path.join(args.vectorizer_save_dir,datetime.strftime(now, '%y%m%d_%H%M%S.vec'))
	file_ = open(file_path, 'ab')
	pickle.dump(serializer, file_)
	print(f"Vectorizer saved: {file_path}")
	

# model
classifier = ai.ReviewClassifier(num_features=len(vectorizer.review_vocab))
classifier = classifier.to(args.device)
# Load the model for inference mode
if not args.train:
	classifier.load_state_dict(torch.load(args.model_state_file))

# loss function and optimizer
loss_func = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)

if args.train:
	ai.train_model(classifier, dataset, loss_func, optimizer, args)
else: # Inference mode
	review_text = args.review
	out = ai.predict_rating(review_text, classifier, vectorizer, decision_threshold=0.5)
	print(f"{review_text} --> {out}")
