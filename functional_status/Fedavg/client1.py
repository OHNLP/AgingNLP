from collections import OrderedDict
import warnings
import os

import flwr as fl
import torch
import numpy as np

import random
from torch.utils.data import DataLoader

from datasets import load_dataset, load_metric

from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import AdamW

from transformers import BertTokenizer
import pandas as pd
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import random
import numpy as np
import time
import datetime
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix
from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer,
                                  DistilBertConfig,
                                  DistilBertForSequenceClassification,
                                  DistilBertTokenizer,
                                  AlbertConfig,
                                  AlbertForSequenceClassification, 
                                  AlbertTokenizer,
                                )
print ('Start..')

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "distilbert-base-uncased"  # transformer model checkpoint


# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

device = torch.device("cpu")
DEVICE = torch.device("cpu")


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def load_data():
	df = pd.read_csv('/infodev1/phi-projects/dhs/workspace-sf/MCI/BERT_data/data/NLP_input/fl/training_rst.txt', delimiter= '\t', header=None, names=['docid', 'sentence', 'label'])
# 	df = df.sample(n=5000, random_state=7)
# 	df = df.drop(df[(df.label != '1') & (df.label != '0')].index)
# 	print(df)
	df = df.dropna()

	sentences = df.sentence.values
	labels = [int(i) for i in df.label.values]
	
# 	print(labels)
	
	# Load the BERT tokenizer.
	print('Loading BERT tokenizer...')
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
# 	tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', do_lower_case=True)
# 	tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-v1.1', do_lower_case=True)		
# 	tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_Discharge_Summary_BERT', do_lower_case=True)			

	max_len = 0

	# For every sentence...
	for sent in sentences:

    	# Tokenize the text and add `[CLS]` and `[SEP]` tokens.
	    input_ids = tokenizer.encode(sent, add_special_tokens=True)

    	# Update the maximum sentence length.
	    max_len = max(max_len, len(input_ids))

# 	    if len(input_ids) > 512:
# 	    	print ('^^^^^^^^^^^^^^')
# 	    	print (sent)
# 	    	print ('^^^^^^^^^^^^^^')
	    	
	print('Max sentence length: ', max_len)	
	
	# Tokenize all of the sentences and map the tokens to thier word IDs.
	input_ids = []
	attention_masks = []
	
	# For every sentence...
	for sent in sentences:
		# `encode_plus` will:
		#   (1) Tokenize the sentence.
		#   (2) Prepend the `[CLS]` token to the start.
		#   (3) Append the `[SEP]` token to the end.
		#   (4) Map tokens to their IDs.
		#   (5) Pad or truncate the sentence to `max_length`
		#   (6) Create attention masks for [PAD] tokens.
		encoded_dict = tokenizer.encode_plus(
							sent,                      # Sentence to encode.
							add_special_tokens = True, # Add '[CLS]' and '[SEP]'
							max_length = 512,           # Pad & truncate all sentences.
 							pad_to_max_length = True,
 							return_attention_mask = True,   # Construct attn. masks.
							return_tensors = 'pt',     # Return pytorch tensors.
						)
		# Add the encoded sentence to the list.    
		input_ids.append(encoded_dict['input_ids'])
		
		# And its attention mask (simply differentiates padding from non-padding).
		attention_masks.append(encoded_dict['attention_mask'])
		
	# Convert the lists into tensors.
	input_ids = torch.cat(input_ids, dim=0)
	attention_masks = torch.cat(attention_masks, dim=0)
	labels = torch.tensor(labels)
	
	# Print sentence 0, now as a list of IDs.
	print('Original: ', sentences[0])
	print('Token IDs:', input_ids[0])	
	
	dataset = TensorDataset(input_ids, attention_masks, labels)

	# Calculate the number of samples to include in each set.
	train_size = int(0.85 * len(dataset))
	val_size = len(dataset) - train_size

	# Divide the dataset by randomly selecting samples.
	train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

	print('{:>5,} training samples'.format(train_size))
	print('{:>5,} validation samples'.format(val_size))



	# The DataLoader needs to know our batch size for training, so we specify it 
	# here. For fine-tuning BERT on a specific task, the authors recommend a batch 
	# size of 16 or 32.
	batch_size = 16

	# Create the DataLoaders for our training and validation sets.
	# We'll take training samples in random order. 
	train_dataloader = DataLoader(
				train_dataset,  # The training samples.
				sampler = RandomSampler(train_dataset), # Select batches randomly
				batch_size = batch_size # Trains with this batch size.
			)

	# For validation the order doesn't matter, so we'll just read them sequentially.
	validation_dataloader = DataLoader(
				val_dataset, # The validation samples.
				sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
				batch_size = batch_size # Evaluate with this batch size.
			)
	return train_dataloader, validation_dataloader
		

def train(model, train_dataloader, epochs):
	optimizer = AdamW(model.parameters(),
					  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
					  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
					)


	# Total number of training steps is [number of batches] x [number of epochs]. 
	# (Note that this is not the same as the number of training samples).
	total_steps = len(train_dataloader) * epochs

	# Create the learning rate scheduler.
	scheduler = get_linear_schedule_with_warmup(optimizer, 
												num_warmup_steps = 0, # Default value in run_glue.py
												num_training_steps = total_steps)



	# This training code is based on the `run_glue.py` script here:
	# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

	# Set the seed value all over the place to make this reproducible.
	seed_val = 42

	random.seed(seed_val)
	np.random.seed(seed_val)
	torch.manual_seed(seed_val)
	torch.cuda.manual_seed_all(seed_val)

	# We'll store a number of quantities such as training and validation loss, 
	# validation accuracy, and timings.
	training_stats = []

	# Measure the total training time for the whole run.
	total_t0 = time.time()

	# For each epoch...
	for epoch_i in range(0, epochs):
	
		# ========================================
		#               Training
		# ========================================
	
		# Perform one full pass over the training set.

		print("")
		print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
		print('Training...')

		# Measure how long the training epoch takes.
		t0 = time.time()

		# Reset the total loss for this epoch.
		total_train_loss = 0

		# Put the model into training mode. Don't be mislead--the call to 
		# `train` just changes the *mode*, it doesn't *perform* the training.
		# `dropout` and `batchnorm` layers behave differently during training
		# vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
		model.train()

		# For each batch of training data...
		for step, batch in enumerate(train_dataloader):

			# Progress update every 40 batches.
			if step % 40 == 0 and not step == 0:
				# Calculate elapsed time in minutes.
				elapsed = format_time(time.time() - t0)
			
				# Report progress.
				print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

			# Unpack this training batch from our dataloader. 
			#
			# As we unpack the batch, we'll also copy each tensor to the GPU using the 
			# `to` method.
			#
			# `batch` contains three pytorch tensors:
			#   [0]: input ids 
			#   [1]: attention masks
			#   [2]: labels 
			b_input_ids = batch[0].to(device)
			b_input_mask = batch[1].to(device)
			b_labels = batch[2].to(device)

			# Always clear any previously calculated gradients before performing a
			# backward pass. PyTorch doesn't do this automatically because 
			# accumulating the gradients is "convenient while training RNNs". 
			# (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
			model.zero_grad()        

			# Perform a forward pass (evaluate the model on this training batch).
			# The documentation for this `model` function is here: 
			# https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
			# It returns different numbers of parameters depending on what arguments
			# arge given and what flags are set. For our useage here, it returns
			# the loss (because we provided labels) and the "logits"--the model
			# outputs prior to activation.
			loss, logits = model(b_input_ids, 
								 token_type_ids=None, 
								 attention_mask=b_input_mask, 
								 labels=b_labels, return_dict=False)

			# Accumulate the training loss over all of the batches so that we can
			# calculate the average loss at the end. `loss` is a Tensor containing a
			# single value; the `.item()` function just returns the Python value 
			# from the tensor.
			total_train_loss += loss.item()

			# Perform a backward pass to calculate the gradients.
			loss.backward()

			# Clip the norm of the gradients to 1.0.
			# This is to help prevent the "exploding gradients" problem.
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

			# Update parameters and take a step using the computed gradient.
			# The optimizer dictates the "update rule"--how the parameters are
			# modified based on their gradients, the learning rate, etc.
			optimizer.step()

			# Update the learning rate.
			scheduler.step()

		# Calculate the average loss over all of the batches.
		avg_train_loss = total_train_loss / len(train_dataloader)            
	
		# Measure how long this epoch took.
		training_time = format_time(time.time() - t0)

		print("")
		print("  Average training loss: {0:.2f}".format(avg_train_loss))
		print("  Training epcoh took: {:}".format(training_time))
		return avg_train_loss

def test(model, validation_dataloader, epochs):
	training_stats = []
	print("")
	print("Running Validation...")
	t0 = time.time()

	# Put the model in evaluation mode--the dropout layers behave differently
	# during evaluation.
	model.eval()
	
	predictions , true_labels = [], []

	# Tracking variables 
	total_eval_accuracy = 0
	total_eval_loss = 0
	nb_eval_steps = 0
	y_pred, y_true = [], []
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

	# Evaluate data for one epoch
	for batch in validation_dataloader:
	
		# Unpack this training batch from our dataloader. 
		#
		# As we unpack the batch, we'll also copy each tensor to the GPU using 
		# the `to` method.
		#
		# `batch` contains three pytorch tensors:
		#   [0]: input ids 
		#   [1]: attention masks
		#   [2]: labels 
		b_input_ids = batch[0].to(device)
		b_input_mask = batch[1].to(device)
		b_labels = batch[2].to(device)
	
		# Tell pytorch not to bother with constructing the compute graph during
		# the forward pass, since this is only needed for backprop (training).
		with torch.no_grad():        

			# Forward pass, calculate logit predictions.
			# token_type_ids is the same as the "segment ids", which 
			# differentiates sentence 1 and 2 in 2-sentence tasks.
			# The documentation for this `model` function is here: 
			# https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
			# Get the "logits" output by the model. The "logits" are the output
			# values prior to applying an activation function like the softmax.
			(loss, logits) = model(b_input_ids, 
								   token_type_ids=None, 
								   attention_mask=b_input_mask,
								   labels=b_labels, return_dict=False)
		
		# Accumulate the validation loss.
		total_eval_loss += loss.item()

		# Move logits and labels to CPU
		logits = logits.detach().cpu().numpy()
		label_ids = b_labels.to('cpu').numpy()

		# Calculate the accuracy for this batch of test sentences, and
		# accumulate it over all batches.
		total_eval_accuracy += flat_accuracy(logits, label_ids)


		# Store predictions and true labels
		predictions.append(logits)
		true_labels.append(label_ids)	
		f1_set = []
		for i in range(len(true_labels)):
			pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
			y_pred += list(pred_labels_i)
			y_true += list(true_labels[i])
		
	cm = confusion_matrix(y_pred, y_true)
	f1 = f1_score(y_pred, y_true)


	# Report the final accuracy for this validation run.
	avg_val_accuracy = total_eval_accuracy / float(len(validation_dataloader))
	print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

	# Calculate the average loss over all of the batches.
	avg_val_loss = total_eval_loss / float(len(validation_dataloader))

	# Measure how long the validation run took.
	validation_time = format_time(time.time() - t0)

	print("  Validation Loss:", avg_val_loss)
	print("  Validation took: {:}".format(validation_time))

	# Record all statistics from this epoch.
	training_stats.append(
		{
			'epoch': epochs + 1,
# 			'Training Loss': avg_train_loss,
			'Valid. Loss': avg_val_loss,
			'Valid. Accur.': avg_val_accuracy,
# 			'Training Time': training_time,
			'Validation Time': validation_time})
	print("")
	print("Training complete!")
	
	output_dir = './output_models/site_rst/bert_output/'	
	# Create output directory if needed
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	print("Saving model to %s" % output_dir)

	# Save a trained model, configuration and tokenizer using `save_pretrained()`.
	# They can then be reloaded using `from_pretrained()`
	model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
	model_to_save.save_pretrained(output_dir)
	tokenizer.save_pretrained(output_dir)
	
	print (cm)
		
	return avg_val_loss, f1 		




def main():
#     net = AutoModelForSequenceClassification.from_pretrained(
#         CHECKPOINT, num_labels=2
#     ).to(DEVICE)

	# Load BertForSequenceClassification, the pretrained BERT model with a single 
	# linear classification layer on top. 
	model = BertForSequenceClassification.from_pretrained(
# 		"emilyalsentzer/Bio_ClinicalBERT",
		"bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
		num_labels = 2, # The number of output labels--2 for binary classification.
						# You can increase this for multi-class tasks.   
		output_attentions = False, # Whether the model returns attentions weights.
		output_hidden_states = False, # Whether the model returns all hidden-states.
	)

	# Tell pytorch to run this model on the GPU.
# 	model.cuda()
	model.to(device)

	# Get all of the model's parameters as a list of tuples.
	params = list(model.named_parameters())

	print('The BERT model has {:} different named parameters.\n'.format(len(params)))
	print('==== Embedding Layer ====\n')
	for p in params[0:5]:
		print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
	print('\n==== First Transformer ====\n')
	for p in params[5:21]:
		print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
	print('\n==== Output Layer ====\n')
	for p in params[-4:]:
		print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


	trainloader, testloader = load_data()

    # Flower client
	class MCIClient1(fl.client.NumPyClient):
		def get_parameters(self, config):
			return [val.cpu().numpy() for _, val in model.state_dict().items()]

		def set_parameters(self, parameters):
			params_dict = zip(model.state_dict().keys(), parameters)
			state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
			model.load_state_dict(state_dict, strict=True)

		def fit(self, parameters, config):
			self.set_parameters(parameters)
			print("Training Started...")
			train(model, trainloader, epochs=4)
			print("Training Finished.")
			return self.get_parameters(config={}), len(trainloader), {}

		def evaluate(self, parameters, config):
			self.set_parameters(parameters)
			loss, f1 = test(model, testloader, epochs=4)
			return float(loss), len(testloader), {"f1_score": float(f1)}

    # Start client
	fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=MCIClient1())


if __name__ == "__main__":
	main()