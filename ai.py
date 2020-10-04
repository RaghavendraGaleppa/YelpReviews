import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import generate_batches
from utils import preprocess_text
from tqdm import tqdm
from datetime import datetime
import os

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

def compute_accuracy(y_pred, target):
	y_pred = torch.sigmoid(y_pred)
	y_pred[y_pred < 0.5] = 0
	y_pred[y_pred >= 0.5] = 1
	accuracy = (y_pred == target).sum().float()/target.size()[0]
	return accuracy.item()

def train_model(classifier, dataset, loss_func, optimizer, args,):
	train_state = make_train_state(args)

	for epoch_index in range(args.num_epochs):
		train_state['epoch_index'] = epoch_index

		dataset.set_split('train')
		batch_generator = generate_batches(dataset,
				batch_size=args.batch_size,
				device=args.device)
		running_loss = 0.0
		running_acc = 0.0
		classifier.train()
		batch_generator = tqdm(batch_generator)
		batch_generator.set_description(f"E:({epoch_index+1}/{args.num_epochs})")

		for batch_index, batch_dict in enumerate(batch_generator):
			optimizer.zero_grad()
			y_pred = classifier(batch_dict['x_data'].float())
			loss = loss_func(y_pred, batch_dict['y_target'].float())
			loss_batch = loss.item()
			running_loss += (loss_batch - running_loss) / (batch_index + 1)

			loss.backward()
			optimizer.step()

			acc_batch = compute_accuracy(y_pred, batch_dict['y_target'])
			running_acc += (acc_batch-running_acc)/(batch_index + 1)
			batch_generator.set_postfix(train_acc=running_acc, train_loss=running_loss)

		train_state['train_acc'].append(running_acc)
		train_state['train_loss'].append(running_loss)

		dataset.set_split('valid')
		batch_generator = generate_batches(dataset,
				batch_size=args.batch_size,
				device=args.device)
		batch_generator = tqdm(batch_generator)
		batch_generator.set_description(f"E:({epoch_index+1}/{args.num_epochs})")
		running_loss = 0.0
		running_acc = 0.0
		classifier.eval()

		for batch_index, batch_dict in enumerate(batch_generator):
			y_pred = classifier(batch_dict['x_data'].float())

			loss = loss_func(y_pred, batch_dict['y_target'].float())
			loss_batch = loss.item()
			running_loss += (loss_batch - running_loss) / (batch_index + 1)

			acc_batch = compute_accuracy(y_pred, batch_dict['y_target'])
			running_acc += (acc_batch-running_acc)/(batch_index + 1)
			batch_generator.set_postfix(val_acc=running_acc, val_loss=running_loss)

		train_state['val_acc'].append(running_acc)
		train_state['val_loss'].append(running_loss)

	if args.save_model:
		if not os.path.isdir(args.model_save_dir):
			os.makedirs(args.model_save_dir)

		now = datetime.now()
		file_path = os.path.join(args.model_save_dir,datetime.strftime(now, '%y%m%d_%H%M%S.pth'))
		torch.save(classifier.state_dict(), file_path)
		
	
def predict_rating(review_text, classifier, vectorizer, decision_threshold=0.5):
	"""
		- Given the classifier, vectorizer and text, classify whether that text is a positive or
		negative review

		args:
			review_text(str): The review that needs to be classified
			classifier(ReviewClassifier): The model that has been trained for classification
			vectorizer(utils.Vectorizer): The Vectorizer that will be used to convert the text to a vector
		returns:
			class(str): The class, which is either positive or negative
			
	"""
	
	review_text = preprocess_text(review_text)
	review_vector = vectorizer.vectorize(review_text)
	result = torch.sigmoid(classifier(review_vector.view(1,-1)))
	class_label = None	
	if result.item() < decision_threshold:
		class_label = 0
	else:
		class_label = 1
		
	return vectorizer.review_vocab.lookup_index(class_label)

