from argument_parser import args
import ai
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


if args.device is None:
	if torch.cuda.is_available():
		args.device = "cuda"
		print(f"Training on GPU")
	else:
		args.device = "cpu"
		print(f"Training on CPU")

# Dataset and Vectorizer
dataset = utils.ReviewDataset.create_and_make_dataset(args)
vectorizer = dataset.get_vectorizer()

# model
classifier = ai.ReviewClassifier(num_features=len(vectorizer.review_vocab))
classifier = classifier.to(args.device)

# loss function and optimizer
loss_func = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)

ai.train_model(classifier, dataset, loss_func, optimizer, args)
