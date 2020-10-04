import re
import numpy as np
import pandas as pd
import string
from collections import Counter
from collections import defaultdict

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def preprocess_text(text):
	"""
		- Clean the data
		- Add whitespaces around punctuation symbols, and remove extraneous symbols
		that arent punctuations.
	"""
	text = text.lower()
	text = re.sub(r"([.,!?])", r" \1 ", text)
	text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
	return text

def convert_stars(stars):
	"""
		- Takes a ratings and converts it into either positive or negative rating
		args:
			stars(int): The stars that restaurant recieved
		returns
			rating(str): Returs rating for that stars, either positive or negative
	"""
	
	if stars > 2:
		return 'positive'
	else:
		return 'negative'

def create_and_load_data(args):
	"""
		- Creates a dataFrame, preprocesses and loads it. Also splits the data in train, test and valid
		args:
			args(ArgumentParser): An argmument parser, which should contain an attribute 
			called data_path that holds the location of the csv
		returns:
			final_reviews(pd.DataFrame): The preprocessed and splitted dataframe
	"""
	review_df = pd.read_csv(args.data_path)
	if 'stars' in review_df.columns:
		review_df.stars = review_df.stars.apply(convert_stars)
		review_df = review_df.rename(columns={'stars':'ratings'})
	if 'text' in review_df.columns:
		review_df = review_df.rename(columns={'text':'review'})

	by_ratings = defaultdict(list)
	for row_index, row in review_df.iterrows():
		by_ratings[row.ratings].append(row.to_dict())

	final_list = []
	np.random.seed(args.seed)

	for ratings, item_list in by_ratings.items():
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

	# Preprocess the reviews
	final_reviews = pd.DataFrame(final_list)
	final_reviews.review = final_reviews.review.apply(preprocess_text)

	return final_reviews

class Vocabulary(object):
	"""
		- This function acts a bijection function, which maps each token to an integer and 
		vice versa.
		- You can add token, lookup its id and use an id to get a token.
		- There is also a special token called the unknow <UNK> that will be used for words that 
		have either not been seen in the training set or occur much less frequently throughout the
		dataset.
	"""

	def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>",):
		"""
			- token_to_idx: a pre-existing mapping of token to idx.
			- add_unk: whether the unknow token id should be added to the vocabulary
			- unk_token: the unknow token
		"""
		if token_to_idx is None:
			token_to_idx = {}

		self._token_to_idx = token_to_idx
		self._idx_to_token = {idx:token for token, idx in self._token_to_idx.items()}

		self._add_unk = add_unk
		self._unk_token = unk_token
		self.unk_index = -1

		if add_unk:
			self.unk_index = self.add_token(unk_token)


	def add_token(self,token,to_lower=True):
		"""
			- This function will add a token to the Vocabulary if, the token is not
			already present in the Vocabulary.
			- If the token is already present, then the function will simply return the
			index to that token

			args:
				token(str): The token that needs to be added to the vocabulary
				to_lower(bool): Flag whether to convert the given token to lower case
			returns:
				index(int): The index of the token in the Vocabulary
		"""
		if to_lower:
			token = token.lower()
		if token in self._token_to_idx:
			index = self._token_to_idx[token]
		else:
			index = len(self._token_to_idx)
			self._token_to_idx[token] = index
			self._idx_to_token[index] = token
		return index

	def lookup_token(self, token, to_lower=True):
		"""
			- Function will return the index to the token, if the token is present in the
			Vocabulary, else will return UNK index

			args:
				token(str): The token that needs to be looked for
				to_lower(bool): Flag whether to convert the given token to lower case
			returns:
				index(int): The index of the token in the Vocabulary
		"""
		if to_lower:
			token = token.lower()
		if self._add_unk:
			return self._token_to_idx.get(token, self.unk_index)
		else:
			return self._token_to_idx[token]


	def lookup_index(self, index):
		"""
			- Function will return the token corresponding to that index in the Vocabulary. If the
			index is invalid, then the function will raise a ValueError

			args:
				index(int): The index of the token that needs to looked for
			returns:
				token(str): The token corresponding to that index 
		"""

		if index not in self._idx_to_token:
			raise ValueError(f"The index {index} is not present in the Vocabulary.")
		return self._idx_to_token[index]

	def to_serializer(self):
		"""
			Return a serializable data structure, with essential components
		"""
		return {'token_to_idx':self._token_to_idx,
				'add_unk': self._add_unk,
				'unk_token': self._unk_token,
				}

	@classmethod
	def from_serializer(cls, contents):
		"""
			- Create and initialize a class with the contents
			returns:
				object(Vocabulary): The Vocabulary that has been initialized
		"""
		return cls(**contents)


	def __str__(self):
		return f"<Vocabulary(size={len(self)})>"

	def __len__(self):
		return len(self._token_to_idx)

class Vectorizer(object):
	"""
		- This class will be used to iterate through each token in an input data point into its integer form. The result
		will be a vector.

		- Since many input data points will be combined to create a mini batch, the length of each vector produced by
		by this class should be of same length.

		- To keep things simple we will use one-hot encoding as the method for vecotrization. The disadvantage is that, 
		the vector will be very sparse and the order in which words occur in the sentence will not be stored.
		
	"""

	def __init__(self, review_vocab, rating_vocab):
		"""
			args:
				review_vocab: maps words to integers
				rating_vocab: maps stars to class labels
		"""
		self.review_vocab = review_vocab
		self.rating_vocab = rating_vocab

	def vectorize(self,review):
		"""
			- Given a token, convert it into its one-hot encoding form
			args:
				review(str): The correspoding review 
			returns:
				vector(ndarray): The one-hot representation of that review
		"""
		vector = np.zeros(shape=(len(self.review_vocab),), dtype=np.float32)

		for token in review.split(" "):
			if token not in string.punctuation:
				index = self.review_vocab.lookup_token(token)
				vector[index] = 1
		return vector

	@classmethod
	def from_dataframe(cls, review_df, cutoff=25):
		"""
			- Return a class object created using the dataframe
			- Assuming that the review_df is already preprocessed

			args:
				cls: Class Vectorizer
				review_df(pd.DataFrame) : The dataframe that needs to be vectorized
				cutoff(int): The minimum count a word needs to make it into the Vocabulary. 
				All other words with count less than cutoff will be marked as unknow tokens
			returns:
				vector(Vectorizer): An instance of Vectorizer class
		"""
		review_vocab = Vocabulary(add_unk=True)
		rating_vocab = Vocabulary(add_unk=False)

		word_counts = Counter()
		for review in review_df.review.tolist():
			for token in review.split(" "):
				word_counts[token] += 1

		# Add ratings to rating vocabulary
		for rating in review_df.ratings.unique().tolist():
			rating_vocab.add_token(rating)

		# Add tokens to review vocabulary
		for token in word_counts:
			if word_counts[token] > cutoff:
				index = review_vocab.add_token(token)

		return cls(review_vocab, rating_vocab)

	def to_serializer(self):
		return {'review_vocab': self.review_vocab.to_serializer(),
				'rating_vocab': self.rating_vocab.to_serializer(),
				}
	
	@classmethod
	def from_serializer(cls, contents):
		review_vocab = Vocabulary.from_serializer(contents['review_vocab'])
		rating_vocab = Vocabulary.from_serializer(contents['rating_vocab'])
		return cls(review_vocab=review_vocab, rating_vocab=rating_vocab)

class ReviewDataset(Dataset):
	def __init__(self, review_df, vectorizer):
		"""
			args:
				review_df(pd.DataFrame): The dataframe for reviews which has already been vectorized
				vectorizer(Vectorizer): The vectorizer instance that has been initialized with 
				the given dataframe.
		"""

		self.review_df = review_df
		self._vectorizer = vectorizer

		self.train_df = self.review_df[self.review_df.split == 'train']
		self.train_size = len(self.train_df)

		self.valid_df = self.review_df[self.review_df.split == 'valid']
		self.valid_size = len(self.valid_df)

		self.test_df = self.review_df[self.review_df.split == 'test']
		self.test_size = len(self.test_df)

		self._lookup_dict = {'train': (self.train_df, self.train_size),
							'valid': (self.valid_df, self.valid_size),
							'test': (self.test_df, self.test_size),
							}

		self.set_split('train')

	def get_vectorizer(self):
		return self._vectorizer

	def set_split(self, split="train"):
		self._target_split = split
		self._target_df, self._target_size = self._lookup_dict[split]

	def __len__(self):
		return self._target_size

	def __getitem__(self, index):
		row = self._target_df.iloc[index]
		review_vector = self._vectorizer.vectorize(row.review)
		rating_vector = self._vectorizer.rating_vocab.lookup_token(row.ratings)

		return {'x_data': review_vector, 'y_target': rating_vector}

	def num_batches(self, batch_size):
		"""
			- Given the batch_size return how many batches can you create from the dataset
		"""

		return len(self) // batch_size

	@classmethod
	def create_and_make_dataset(cls, args):
		review_df = create_and_load_data(args)
		vectorizer = Vectorizer.from_dataframe(review_df)
		return cls(review_df, vectorizer)

def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cpu"):
	"""
		A Generator function that wraps the Pytorch DataLoader class. 
	"""

	dataloader = DataLoader(dataset, batch_size=batch_size, 
			shuffle=shuffle, drop_last=drop_last)


	for data_dict in dataloader:
		out_data_dict = {}
		for name, tensor in data_dict.items():
			out_data_dict[name] = data_dict[name].to(device)

		yield out_data_dict
