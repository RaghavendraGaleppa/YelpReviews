import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReviewClassifier(nn.Module):
	"""
		- A simple perceptron classifier
	"""

	def __init__(self, num_features):
		super(ReviewClassifier, self).__init__()
		self.fc1 = nn.Linear(in_features=num_features, out_features=1)

	def forward(self, x, apply_sigmoid=False):
		x = self.fc1(x)
		if apply_sigmoid:
			x = F.sigmoid(x)

		return x.squeeze()

def make_train_state(args):
	return {'epoch_index': 0,
			'train_loss':[],
			'train_acc':[],
			'val_loss':[],
			'val_acc':[],
			'test_loss':-1,
			'test_acc':-1
			}

