# Training

import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch as torch
import wandb
from sklearn.metrics import f1_score
import csv
# Save and Load Functions

def save_checkpoint(save_path, model, optimizer, valid_loss):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, optimizer):

    if load_path==None:
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):

    if load_path==None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

# Training Function

def train(model,
          optimizer,
          criterion = nn.BCELoss(),
          train_loader = None,
          valid_loader = None,
          num_epochs = 5,
          eval_every = None,
          file_path = None,
          best_valid_loss = float("Inf"),
          device = None
          ):
    
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        for (docid,  (titletext, text_len), labels), _ in train_loader:           
            labels = labels.to(device)
            titletext = titletext.to(device)
            titletext_len = text_len.to(device)
            output = model(titletext, titletext_len)

            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():                    
                  # validation loop
                  for (docid,  (titletext, text_len), labels), _ in valid_loader:
                      labels = labels.to(device)
                      titletext = titletext.to(device)
                      titletext_len = text_len.to(device)
                      output = model(titletext, titletext_len)

                      loss = criterion(output, labels)
                      valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_valid_loss))
                              
                wandb.log(
						(
							{
								"train_loss": average_train_loss,
								"valid_loss": average_valid_loss,
							}
						)
						)                                
                
                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/model.pt', model, optimizer, best_valid_loss)
                    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    
    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')



# Evaluation Function

def evaluate(model, test_loader, version='title', threshold=0.5):
	y_pred = []
	y_true = []
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model.eval()
	with open('/Fall/output/output_lstm.txt', 'w') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter='\t')
		with torch.no_grad():
			for (docid,  (titletext, text_len), labels), _ in test_loader:           
				labels = labels.to(device)
				titletext = titletext.to(device)
				titletext_len = text_len.to(device)
				output = model(titletext, titletext_len)

				output = (output > threshold).int()
				y_pred.extend(output.tolist())
				y_true.extend(labels.tolist())
							
				spamwriter.writerow([docid[0], output.tolist()[0], labels.tolist()[0]])
            

	print('Classification Report:')
	cm = f1_score(y_true, y_pred)
	print (cm)


    

