import flwr as fl

from typing import Dict, Optional, Tuple, List
from pathlib import Path

import csv

from flwr.common import Metrics

from collections import OrderedDict

import torch
import numpy as np

import random
from torch.utils.data import DataLoader


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
from flwr.server.strategy import dpfedavg_adaptive

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




def get_model_params(model):
    """Returns a model's parameters."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]



def test(net, testloader, steps: int = None, device: str = "cpu"):
    """Validate the network on the entire test set."""
    print("Starting evalutation...")
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if steps is not None and batch_idx == steps:
                break
    if steps is None:
        loss /= len(testloader.dataset)
    else:
        loss /= total
    accuracy = correct / total
    net.to("cpu")  # move model back to CPU
    return loss, accuracy


model = BertForSequenceClassification.from_pretrained(
# 		"emilyalsentzer/Bio_ClinicalBERT",
	"bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
	num_labels = 2, # The number of output labels--2 for binary classification.
					# You can increase this for multi-class tasks.   
	output_attentions = False, # Whether the model returns attentions weights.
	output_hidden_states = False, # Whether the model returns all hidden-states.
)

model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

# Tell pytorch to run this model on the GPU.
model.cuda()

def get_evaluate_fn(model: torch.nn.Module):
	"""Return an evaluation function for server-side evaluation."""

	df = pd.read_csv('/data/sfu3/workspace/mci/data/training_mchs.csv', delimiter='\t', header=None, names=['docid', 'sentence', 'label'])
# 	df = df.sample(n=5000, random_state=5)
	df = df.dropna()
	sentences, labels, docid = [],[],[]
	print(df)
	print('check')
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
# 		sents.append(encoded_dict['sent'])
	
		# And its attention mask (simply differentiates padding from non-padding).
		attention_masks.append(encoded_dict['attention_mask'])

	# Convert the lists into tensors.
	input_ids = torch.cat(input_ids, dim=0)
	attention_masks = torch.cat(attention_masks, dim=0)
	labels = torch.tensor(labels)

	# Set the batch size.  
	batch_size = 8

	# Create the DataLoader.
	prediction_data = TensorDataset(input_ids, attention_masks, labels)
	prediction_sampler = SequentialSampler(prediction_data)
	prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    # The `evaluate` function will be called after every round
	def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
		# Update model with the latest parameters
		params_dict = zip(model.state_dict().keys(), parameters)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		model.load_state_dict(state_dict, strict=True)		

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
		f1_set = []
		matthews_set = []
		y_pred, y_true = [], []
		# Evaluate each test batch using Matthew's correlation coefficient
		print('Calculating Matthews Corr. Coef. for each batch...')
		# For each input batch...
		for i in range(len(true_labels)):
			pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
			y_pred += list(pred_labels_i)
			y_true += list(true_labels[i])
		f1 = f1_score(y_pred, y_true)  
		
		with open('bertbase_eval_fl.csv', 'w') as csvfile:
			spamwriter = csv.writer(csvfile, delimiter='\t')
			for y in range(len(y_pred)):
		# 			if y_pred[y] != y_true[y]:
		# 			print (docid[y],'\t',sentences[y],'\t',y_true[y],'\t',y_pred[y])
				spamwriter.writerow([docid[y], sentences[y],y_true[y],y_pred[y]])
		
		return f1, {"f1_score": f1}
	
	return evaluate

def fit_config(server_round: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 16,
        "local_epochs": 1 if server_round < 2 else 2,
    }
    return config

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracy = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"f1_score": sum(accuracy) / sum(examples)}
    
    
def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}	


if __name__ == "__main__":

    # FL eval
# 	strategy = fl.server.strategy.FedAvg(
# 	fraction_fit=1.0,
# 	fraction_evaluate=1.0)

	# Central eval
	strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
	fraction_evaluate=1.0,
	evaluate_fn=get_evaluate_fn(model),
	on_fit_config_fn=fit_config,
	on_evaluate_config_fn=evaluate_config,
	initial_parameters=fl.common.ndarrays_to_parameters(model_parameters)
# 	fit_metrics_aggregation_fn = True,
# 	evaluate_metrics_aggregation_fn = weighted_average
)

	# Implementing Fixed and Adaptive DF strategies
	
	# strategy = dpfedavg_adaptive.DPFedAvgFixed(
	# 	strategy=strategy,
	# 	# clip_norm=conf['dp_clipnorm'],
	# 	num_sampled_clients=3,
	# 	noise_multiplier=0.1,
	# )

	strategy = dpfedavg_adaptive.DPFedAvgAdaptive(
		strategy=strategy,
		# clip_norm=conf['dp_clipnorm'],
		num_sampled_clients=3,
		noise_multiplier=0.1,
	)

	# Start server
	fl.server.start_server(
		server_address="0.0.0.0:8080",
		config=fl.server.ServerConfig(num_rounds=3),
		strategy=strategy,
		grpc_max_message_length = 995896761
	# 		num_clients=NUM_CLIENTS,
	)
	
	
	
	
	