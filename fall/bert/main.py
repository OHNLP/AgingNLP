# @article{Wolf2019HuggingFacesTS,
#   title={HuggingFace's Transformers: State-of-the-art Natural Language Processing},
#   author={Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush},
#   journal={ArXiv},
#   year={2019},
#   volume={abs/1910.03771}
# }

import csv
import re
import os
import glob
from shutil import copyfile
from bs4 import BeautifulSoup 
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import torch
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
# If there's a GPU available...
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def rad_parser(line):
	line = str(line.encode('utf-8'))
	line = line.replace('[', '')
	line = line.replace(']', '')
	line = line.replace('\'', '')
	line = line.replace('}', '')
	# line = line.replace('', '')
	line = line[121:]
	line = line.split('\par')
	line_str = ''
	for m in line:
		line_str += m + '\n' 
	line = line_str
	return line

# 
# Training start
def train():
	epochs_num = 2
	# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
	output_dir = './post_trained_model/'
	df = pd.read_csv('../../data/training.txt', delimiter= '|', header=None, names=['docid', 'sentence', 'label'])
# 	df = df.sample(n=1500, random_state=5)
# 	print (df)
	sentences = df.sentence.values
	labels = [int(i) for i in df.label.values]
	
	# Load the BERT tokenizer.
	print('Loading BERT tokenizer...')
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

	max_len = 0

	# For every sentence...
	for sent in sentences:
    	# Tokenize the text and add `[CLS]` and `[SEP]` tokens.
	    input_ids = tokenizer.encode(sent, add_special_tokens=True)
    	# Update the maximum sentence length.
	    max_len = max(max_len, len(input_ids))
	print('Max sentence length: ', max_len)	
	
	# Tokenize all of the sentences and map the tokens to thier word IDs.
	input_ids = []
	attention_masks = []
	# For every sentence...
	for sent in sentences:
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
	# Load BertForSequenceClassification, the pretrained BERT model with a single 
	# linear classification layer on top. 
	model = BertForSequenceClassification.from_pretrained(
		"bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
		num_labels = 2, # The number of output labels--2 for binary classification.
						# You can increase this for multi-class tasks.   
		output_attentions = False, # Whether the model returns attentions weights.
		output_hidden_states = False, # Whether the model returns all hidden-states.
	)

	# Tell pytorch to run this model on the GPU.
	model.cuda()
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

	# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
	# I believe the 'W' stands for 'Weight Decay fix"
	optimizer = AdamW(model.parameters(),
					  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
					  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
					)

	# Number of training epochs. The BERT authors recommend between 2 and 4. 
	# We chose to run for 4, but we'll see later that this may be over-fitting the
	# training data.
	epochs = epochs_num

	# Total number of training steps is [number of batches] x [number of epochs]. 
	# (Note that this is not the same as the number of training samples).
	total_steps = len(train_dataloader) * epochs

	# Create the learning rate scheduler.
	scheduler = get_linear_schedule_with_warmup(optimizer, 
												num_warmup_steps = 0, # Default value in run_glue.py
												num_training_steps = total_steps)


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

		print("")
		print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
		print('Training...')

		# Measure how long the training epoch takes.
		t0 = time.time()

		# Reset the total loss for this epoch.
		total_train_loss = 0

		model.train()

		# For each batch of training data...
		for step, batch in enumerate(train_dataloader):

			# Progress update every 40 batches.
			if step % 40 == 0 and not step == 0:
				# Calculate elapsed time in minutes.
				elapsed = format_time(time.time() - t0)
			
				# Report progress.
				print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

			#
			# `batch` contains three pytorch tensors:
			#   [0]: input ids 
			#   [1]: attention masks
			#   [2]: labels 
			b_input_ids = batch[0].to(device)
			b_input_mask = batch[1].to(device)
			b_labels = batch[2].to(device)

			model.zero_grad()        

			loss, logits = model(b_input_ids, 
								 token_type_ids=None, 
								 attention_mask=b_input_mask, 
								 labels=b_labels)

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
		print("")
		print("Running Validation...")
		t0 = time.time()

		# Put the model in evaluation mode--the dropout layers behave differently
		# during evaluation.
		model.eval()

		# Tracking variables 
		total_eval_accuracy = 0
		total_eval_loss = 0
		nb_eval_steps = 0

		# Evaluate data for one epoch
		for batch in validation_dataloader:

			b_input_ids = batch[0].to(device)
			b_input_mask = batch[1].to(device)
			b_labels = batch[2].to(device)
		
			with torch.no_grad():        
				(loss, logits) = model(b_input_ids, 
									   token_type_ids=None, 
									   attention_mask=b_input_mask,
									   labels=b_labels)
			
			# Accumulate the validation loss.
			total_eval_loss += loss.item()

			# Move logits and labels to CPU
			logits = logits.detach().cpu().numpy()
			label_ids = b_labels.to('cpu').numpy()

			# Calculate the accuracy for this batch of test sentences, and
			# accumulate it over all batches.
			total_eval_accuracy += flat_accuracy(logits, label_ids)
		

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
				'epoch': epoch_i + 1,
				'Training Loss': avg_train_loss,
				'Valid. Loss': avg_val_loss,
				'Valid. Accur.': avg_val_accuracy,
				'Training Time': training_time,
				'Validation Time': validation_time
			}
		)

	print("")
	print("Training complete!")
	print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
		


	# Create output directory if needed
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	print("Saving model to %s" % output_dir)

	# Save a trained model, configuration and tokenizer using `save_pretrained()`.
	# They can then be reloaded using `from_pretrained()`
	model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
	model_to_save.save_pretrained(output_dir)
	tokenizer.save_pretrained(output_dir)

	# Good practice: save your training arguments together with the trained model
	# torch.save(args, os.path.join(output_dir, 'training_args.bin'))
	
	# Display floats with two decimal places.
	pd.set_option('precision', 2)
	# Create a DataFrame from our training statistics.
	df_stats = pd.DataFrame(data=training_stats)
	# Use the 'epoch' as the row index.
	df_stats = df_stats.set_index('epoch')
	# A hack to force the column headers to wrap.
	#df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])
	# Display the table.
	print (df_stats)
	
	

# Eval start
def reload_saved_model():
	output_dir = './model_save'
	test_dir = './test.txt'
	BATCHSIZE = 32
	
	MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    'albert': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer)
}
	
	config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']	
	
	# Load a trained model and vocabulary that you have fine-tuned
	model = model_class.from_pretrained(output_dir)
	tokenizer = tokenizer_class.from_pretrained(output_dir)

	# Copy the model to the GPU.
	model.to(device)
	
	######### Load test set
	# Load the dataset into a pandas dataframe.
	df = pd.read_csv(test_dir, delimiter= '|', header=None, names=['docid', 'sentence', 'label'])
	sentences, labels, docid = [],[],[]
	
	# Create sentence and label lists
	for index, r in df.iterrows():
		if r['label'] == '1' or r['label'] == '0' or r['label'] == 1 or r['label'] == 0:
			sentences += [r['sentence']]
			labels += [int(r['label'])]	
			docid += [r['docid']]
	# Report the number of sentences.
	print('Number of test sentences: {:,}\n'.format(len(sentences)))

	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
	# Tokenize all of the sentences and map the tokens to thier word IDs.
	input_ids = []
	attention_masks = []
	sents = []

	# For every sentence...
	for sent in sentences:
		encoded_dict = tokenizer.encode_plus(
							sent,                      # Sentence to encode.
							add_special_tokens = True, # Add '[CLS]' and '[SEP]'
							max_length = 218,           # Pad & truncate all sentences.
							pad_to_max_length = True,
							return_attention_mask = True,   # Construct attn. masks.
							return_tensors = 'pt',     # Return pytorch tensors.
					   )
		# Add the encoded sentence to the list.    
		input_ids.append(encoded_dict['input_ids'])
# 		sents.append(encoded_dict['sent'])
	
		# And its attention mask (simply differentiates padding from non-padding).
		attention_masks.append(encoded_dict['attention_mask'])

	# Convert the lists into tensors.
	input_ids = torch.cat(input_ids, dim=0)
	attention_masks = torch.cat(attention_masks, dim=0)
	labels = torch.tensor(labels)

	# Set the batch size.  
	batch_size = BATCHSIZE

	# Create the DataLoader.
	prediction_data = TensorDataset(input_ids, attention_masks, labels)
	prediction_sampler = SequentialSampler(prediction_data)
	prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
		
	######### Prediction on test set	
	print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))
	# Put model in evaluation mode
	model.eval()
	# Tracking variables 
	predictions , true_labels = [], []
	# Predict 
	for batch in prediction_dataloader:
	  # Add batch to GPU
	  batch = tuple(t.to(device) for t in batch)
	  # Unpack the inputs from our dataloader
	  b_input_ids, b_input_mask, b_labels = batch
	  # Telling the model not to compute or store gradients, saving memory and 
	  # speeding up prediction
	  with torch.no_grad():
		  # Forward pass, calculate logit predictions
		  outputs = model(b_input_ids, token_type_ids=None, 
						  attention_mask=b_input_mask)
	  logits = outputs[0]
	  # Move logits and labels to CPU
	  logits = logits.detach().cpu().numpy()
	  label_ids = b_labels.to('cpu').numpy()
	  # Store predictions and true labels
	  predictions.append(logits)
	  true_labels.append(label_ids)
	print('    DONE.')	

	f1_set = []
	matthews_set = []
	y_pred, y_true = [], []
	# Evaluate each test batch using Matthew's correlation coefficient
	print('Calculating Matthews Corr. Coef. for each batch...')
	# For each input batch...
	for i in range(len(true_labels)):
	  # The predictions for this batch are a 2-column ndarray (one column for "0" 
	  # and one column for "1"). Pick the label with the highest value and turn this
	  # in to a list of 0s and 1s.
	  pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
	  y_pred += list(pred_labels_i)
	  y_true += list(true_labels[i])
	  
	print ('docid\tsent\ttrue\tpred\t')
	print (len(y_pred), len(y_true))
	with open('train_fpn.csv', 'w') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter='\t')
		for y in range(len(y_pred)):
			spamwriter.writerow([docid[y], sentences[y],y_true[y],y_pred[y]])

	matthews = matthews_corrcoef(y_pred, y_true)                	
	f1 = f1_score(y_pred, y_true)  
	cm = confusion_matrix(y_pred, y_true)
	print('f1:', f1, 'mcc:', matthews)		
	print (cm)
	
	
	

if __name__ == '__main__':
 
	train()
	# reload_saved_model()