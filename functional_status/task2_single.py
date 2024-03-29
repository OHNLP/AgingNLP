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

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")



	
	
	

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


# import nltk
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

def bs4():
	#check report type
	label1, label2, label3 = '', '', ''
	list_of_files = glob.glob('*.xml')
	txt  = ''
	for indir in list_of_files:
		fp = open(indir).read()
		soup = BeautifulSoup(fp,'xml')
		for item in soup.find_all('THEAD'):
			print (item.text)

def parse_encounter_time(encounter_date):
	encounter_date = encounter_date.split('T')[0]
	encounter_date = encounter_date[:4] + '-' + encounter_date[4:6] + '-' + encounter_date[6:]
	encounter_date = datetime.strptime(encounter_date, '%Y-%m-%d')
	return encounter_date
	

def read_file_list(indir, d):
	opt_notes = []
	with open(indir, 'rU') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=d)
		for row in spamreader:
			try:
				row = [entry.encode("cp1252") for entry in row]
				opt_notes += [row]
			except:
				pass

	return opt_notes

def write_file_list(outdir, df):
	with open(outdir, 'w') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter=',')
		for m in df:
			spamwriter.writerow([m])

def read_file_dict(indir, k, v, d):
	opt_notes = {}
	with open(indir, 'rU') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=d)
		for row in spamreader:
			opt_notes[row[k]] = [row[v]]
	return opt_notes

def write_txt():
	with open("Output.txt", "w") as text_file:
	    text_file.write("Purchase Amount: %s" % TotalAmount)

def match_surg_date():			
	sw = read_file_dict('nlp_approach.csv', 0, 5)
	sf = read_file_list('search_result.csv')
	outlist=[]
	for m in sf:
		if sw[m[0]][0] == m[2]:
			outlist += ['{'.join(m)]
	write_file_list('filtered_.csv', outlist)

def datatime():
	rev_date = datetime.datetime.strptime(rev_date, '%Y-%m-%d %X')

def copy():
	copyfile(src, dst)

def mcn_length(mcn):
	mcn[0] = ('000000'+mcn[0])[-8:]
	return mcn

def read_txt(indir):
	f = open(indir,'r', encoding='cp1252')
	txt = f.read()
	f.close()
	return txt



# 
# Training start

# 
def load_data():
	
	epochs_num = 3
	# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
# 	output_dir = './model_save_epo2_doclevel_10vali_revised_training/'

#   previous best model
# 	output_dir = './model_save_epo4_doclevel_10vali_revised_training/'	

#   Donna revised the training data
	output_dir = './bert-base-mchs-impaired/'	
	
	
# 	df = pd.read_csv('/infodev1/phi-projects/dhs/workspace-sf/MCI/BERT_data/data/NLP_input/fl/training_rst.txt', delimiter= '\t', header=None, names=['docid', 'sentence', 'label'])
	df = pd.read_csv('/infodev1/phi-projects/dhs/workspace-sf/MCI/BERT_data/data/NLP_input/fl/training_mchs.txt', delimiter= '\t', header=None, names=['docid', 'sentence', 'label'])	
# 	df = df.sample(n=1500, random_state=5)
# 	print (df)
# 	df = df.drop(df[(df.label != '1') & (df.label != '0')].index)
# 	print (df)
	sentences = df.sentence.values
	labels = [int(i) for i in df.label.values]
	
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
	batch_size = 32

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
# 		"emilyalsentzer/Bio_ClinicalBERT",
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
# 								 labels=b_labels)

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
		
		# ========================================
		#               Validation
		# ========================================
		# After the completion of each training epoch, measure our performance on
		# our validation set.

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
# 	pd.set_option('precision', 2)
	# Create a DataFrame from our training statistics.
	df_stats = pd.DataFrame(data=training_stats)
	# Use the 'epoch' as the row index.
	df_stats = df_stats.set_index('epoch')
	# A hack to force the column headers to wrap.
	#df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])
	# Display the table.
	print (df_stats)
	
	
# 
# Eval start

# 	

def reload_saved_model():

	output_dir = '/infodev1/phi-projects/dhs/workspace-sf/MCI/BERT_Model/bert-base-mchs-impaired'
	
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
	model = BertForSequenceClassification.from_pretrained(output_dir)
	tokenizer = tokenizer_class.from_pretrained(output_dir)

	# Copy the model to the GPU.
	model.to(device)
	
	######### Load test set
	# Load the dataset into a pandas dataframe.
# 	sites = ['mchs', 'ut', 'rep', 'rst']
# 	for s in sites:
# 		print('######', s)
# 		df = pd.read_csv('/infodev1/phi-projects/dhs/workspace-sf/MCI/BERT_data/data/NLP_input/fl/test_'+s+'.txt', delimiter= '\t', header=None, names=['docid', 'sentence', 'label'])			
# 			
	df = pd.read_csv('/infodev1/phi-projects/dhs/workspace-sf/MCI/BERT_data/data/NLP_input/fl/test_combined2.txt', delimiter= '\t', header=None, names=['docid', 'sentence', 'label'])
	
	print(df)
	df = df.dropna()
	# 	df = df.sample(n=1500, random_state=5)
	sentences, labels, docid = [],[],[]

	# Create sentence and label lists
	for index, r in df.iterrows():
	# 		if r['label'] == '1' or r['label'] == '0' or r['label'] == 1 or r['label'] == 0:
		sentences += [r['sentence']]
		labels += [int(r['label'])]	
		docid += [r['docid']]
	# Report the number of sentences.
	print('Number of test sentences: {:,}\n'.format(len(sentences)))

	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
	# 	tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', do_lower_case=True)	

	# Tokenize all of the sentences and map the tokens to thier word IDs.
	input_ids = []
	attention_masks = []
	sents = []

	# For every sentence...
	for sent in sentences:
		# `encode_plus` will:
		#   (1) Tokenize the sentence.
		#   (2) Prepend the `[CLS]` token to the start.
		#   (3) Append the `[SEP]` token to the end.
		#   (4) Map tokens to their IDs.
		#   (5) Pad or truncate the sentence to `max_length`
		#   (6) Create attention masks for [PAD] tokens.
		encoded_dict =  tokenizer.encode_plus(
							sent,                      # Sentence to encode.
							add_special_tokens = True, # Add '[CLS]' and '[SEP]'
							max_length = 512,           # Pad & truncate all sentences.
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
	batch_size = 32  

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
		  outputs = model(b_input_ids, 
		  				  token_type_ids=None, 
						  attention_mask=b_input_mask,
						  return_dict=False)
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
	  # Calculate and store the coef for this batch. 
	# 	  print (list(pred_labels_i), list(true_labels[i]))

	  y_pred += list(pred_labels_i)
	  y_true += list(true_labels[i])

	print ('docid\tsent\ttrue\tpred\t')
	print (len(y_pred), len(y_true))
	with open('bertbase_test_single_mchs.csv', 'w') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter='\t')
		for y in range(len(y_pred)):
	# 			if y_pred[y] != y_true[y]:
	# 			print (docid[y],'\t',sentences[y],'\t',y_true[y],'\t',y_pred[y])
			spamwriter.writerow([docid[y], sentences[y],y_true[y],y_pred[y]])

# 	matthews = matthews_corrcoef(y_pred, y_true)                	
# 	f1 = f1_score(y_pred, y_true)  
# 	cm = confusion_matrix(y_pred, y_true)
# 	tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
# 	print ('tn, fp, fn, tp')
# 	print (tn, fp, fn, tp)		
# 	print('f1:', f1, 'mcc:', matthews)		
# 	print (cm)
	print('######<FINISHED>')

	
	
def evaluate(dataloader_val) :
    
    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val :
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids': batch[0],
                 'attention_mask': batch[1],
                 'labels': batch[2]}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()
        
        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
        
    loss_val_avg = loss_val_total / len(dataloader_val)
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    
    return loss_val_avg, predictions, true_vals
    	
def run_eval2():
	model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
														  num_labels=len(label_dict),
														  output_attentions=False,
														  output_hidden_states=False)

	model.to(device)

	model.load_state_dict(torch.load('data_volume/finetuned_BERT_epoch_3.model', map_location=torch.device('cpu')))

	_, predictions, true_vals = evaluate(dataloader_validation)
	accuracy_per_class(predictions, true_vals)
	print('f1:', f1_score_func(predictions, true_vals), 'mcc:', matthews_corrcoef_func(predictions, true_vals))
	cf = confusion_matrix_func(predictions, true_vals)
	print (cf)	
	



# load_data()
reload_saved_model()
# 