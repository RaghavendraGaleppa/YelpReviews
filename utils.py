import re

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

class Vocabulary(object):
	"""
		- This function acts a bijection function, which maps each token to an integer and 
		vice versa.
		- You can add token, lookup its id and use an id to get a token.
		- There is also a special token called the unknow <UNK> that will be used for words that 
		have either not been seen in the training set or occur much less frequently throughout the
		dataset.
	"""

	def __init__(self, token_to_idx=None,
			add_unk=True,
			unk_token="<UNK>",
			):
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

	def __str__(self):
		return f"<Vocabulary(size={len(self)})>"

	def __len__(self):
		return len(self._token_to_idx)
