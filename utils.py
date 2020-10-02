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
			unk_token="<UNK>"):
		"""
			- token_to_idx: a pre-existing mapping of token to idx.
			- add_unk: whether the unknow token id should be added to the vocabulary
			- unk_token: the unknow token
		"""
		if token_to_idx is None:
			token_to_idx = {}

		self._token_to_idx = token_to_idx
		self._idx_to_token = {idx:token for token, idx in self._token_to_idx.items()}
