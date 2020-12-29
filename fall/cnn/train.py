import os
import sys
import csv
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import wandb
from sklearn.metrics import f1_score


def train(train_iter, dev_iter, model, args):
    print('strat training')
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_acc = 0
    last_step = 0
    for epoch in range(1, args.epochs+1):
        training_loss = 0
        train_total = 0
        print ('epoch', epoch)
        for batch in train_iter:
            model.train()
            feature, target = batch.text, batch.label
            feature.t_(), target.sub_(1)  # batch first, index align
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()            
            training_loss += loss.item()
            train_total += target.size(0)
            
            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                sys.stdout.write(
                    '\rEpoch[{}] - \rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(epoch, steps, 
                                                                             loss.item(), 
                                                                             accuracy.item(),
                                                                             corrects.item(),
                                                                             batch.batch_size))
                
            if steps % args.test_interval == 0:
                avg_loss, dev_acc = eval(dev_iter, model, args)
  
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    wandb.run.summary["best_accuracy"] = best_acc
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)
        wandb.log(
				(
					{
						"train_loss": training_loss/train_total,
						"valid_loss": avg_loss,
					}
				)
			    )  
        print ('epoch', epoch, steps)			             
            


def eval(data_iter, model, args):
	model.eval()
	corrects, avg_loss, vali_total = 0, 0, 0
	y_true, y_pred = [], []
	with open('/infodev1/phi-projects/dhs/workspace-sf/Fall/output/output2.txt', 'w') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter='\t')
		for batch in data_iter:
			feature, target = batch.text, batch.label
			doc_id = batch.docid
			feature.t_(), target.sub_(1)  # batch first, index align
			if args.cuda:
				feature, target = feature.cuda(), target.cuda()
		
			logit = model(feature)
			loss = F.cross_entropy(logit, target, size_average=False)
			vali_total += target.size(0)
			avg_loss += loss.item()
			corrects += (torch.max(logit, 1)
						 [1].view(target.size()).data == target.data).sum()
			prediction = (torch.max(logit, 1)[1].view(target.size()).data.cpu().numpy()[0])
			y_pred += [prediction]
			y_true += [target.data.cpu().numpy()[0]]
# 			print (doc_id[0], prediction, target.data.cpu().numpy()[0])
			spamwriter.writerow([doc_id[0], prediction, target.data.cpu().numpy()[0]])				 
# 	print (f1_score(y_pred, y_true))

	
	size = len(data_iter.dataset)
	avg_loss /= size
	accuracy = 100.0 * corrects/size
	print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
																	   accuracy, 
																	   corrects, 
																	   size))
	return avg_loss/vali_total, accuracy


def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.item()+1]


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
