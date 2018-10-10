import numpy as np
import pandas as pd
from os import listdir
import random
from nltk import ngrams
import spacy
import string
import pickle as pkl
from collections import Counter
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# save index 0 for unk and 1 for pad
PAD_IDX = 0
UNK_IDX = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_SENTENCE_LENGTH = 200


def build_vocab(all_tokens,max_vocab_size):
    # Returns:
    # id2token: list of tokens, where id2token[i] returns token that corresponds to token i
    # token2id: dictionary where keys represent tokens and corresponding values represent indices
    token_counter = Counter(all_tokens)
    vocab, count = zip(*token_counter.most_common(max_vocab_size))
    id2token = list(vocab)
    token2id = dict(zip(vocab, range(2,2+len(vocab)))) 
    id2token = ['<pad>', '<unk>'] + id2token
    token2id['<pad>'] = PAD_IDX 
    token2id['<unk>'] = UNK_IDX
    return token2id, id2token

def token2index_dataset(tokens_data,token2id,id2token):
    indices_data = []
    for tokens in tokens_data:
        index_list = [token2id[token] if token in token2id else UNK_IDX for token in tokens]
        indices_data.append(index_list)
    return indices_data


class NewsGroupDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """
    
    def __init__(self, data_list, target_list):
        """
        @param data_list: list of newsgroup tokens 
        @param target_list: list of newsgroup targets 

        """
        self.data_list = data_list
        self.target_list = target_list
        assert (len(self.data_list) == len(self.target_list))

    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        
        token_idx = self.data_list[key][:MAX_SENTENCE_LENGTH]
        label = self.target_list[key]
        return [token_idx, len(token_idx), label]

def newsgroup_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all 
    data have the same length
    """
    data_list = []
    label_list = []
    length_list = []
    #print("collate batch: ", batch[0][0])
    #batch[0][0] = batch[0][0][:MAX_SENTENCE_LENGTH]
    for datum in batch:
        label_list.append(datum[2])
        length_list.append(datum[1])
    # padding
    for datum in batch:
        padded_vec = np.pad(np.array(datum[0]), 
                                pad_width=((0,MAX_SENTENCE_LENGTH-datum[1])), 
                                mode="constant", constant_values=0)
        data_list.append(padded_vec)
    return [torch.from_numpy(np.array(data_list)), torch.LongTensor(length_list), torch.LongTensor(label_list)]




class BagOfWords(nn.Module):
    """
    BagOfWords classification model
    """
    def __init__(self, vocab_size, emb_dim,out_dim):
        """
        @param vocab_size: size of the vocabulary. 
        @param emb_dim: size of the word embedding
        """
        super(BagOfWords, self).__init__()
        # pay attention to padding_idx 
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.linear = nn.Linear(emb_dim,out_dim)
    
    def forward(self, data, length):
        """
        
        @param data: matrix of size (batch_size, max_sentence_length). Each row in data represents a 
            review that is represented using n-gram index. Note that they are padded to have same length.
        @param length: an int tensor of size (batch_size), which represents the non-trivial (excludes padding)
            length of each sentences in the data.
        """
        out = self.embed(data)
        out = torch.sum(out, dim=1)
        out /= length.view(length.size()[0],1).expand_as(out).float()
     
        # return logits
        out = self.linear(out.float())
        return out


# Function for testing the model
def test_model(loader, model):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    correct = 0
    total = 0
    model.eval()
    for data, lengths, labels in loader:
        data_batch, length_batch, label_batch = data.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
        outputs = F.softmax(model(data_batch, length_batch), dim=1)
        predicted = outputs.max(1, keepdim=True)[1]
        predicted = predicted.to('cpu')
        total += labels.size(0)
        correct += predicted.eq(labels.data.view_as(predicted)).sum().item()
    return (100 * correct / total)




batch_size = [64]
lrs = [0.01,0.001]
num_grams = [1,2,3,4]
vocab_sizes = [10000,20000,50000,100000]
emb_dims = [100,256]
optims = ['adam','sgd']
somelists = [batch_size,optims,lrs,num_grams,vocab_sizes,emb_dims]

result = list(itertools.product(*somelists))
df_param = pd.DataFrame(result,columns=['batch_size','optimizer','lrs','num_grams','vocab_sizes','emb_dims'])
df_param['train_loss'] = None
df_param['val_accs'] = None
df_param['best_val_acc'] = None
df_param['epoch_best_acc'] = None


for i in range(len(df_param)):
	print(df_param.iloc[i])
	train_data_tokens = pkl.load(open("/scratch/kyt237/NLP/HW1/pickle_no_sw/kyt237/train_data_tokens_grams_{}_no_sw.p".format(df_param.iloc[i]['num_grams']), "rb"))
	all_train_tokens = pkl.load(open("/scratch/kyt237/NLP/HW1/pickle_no_sw/kyt237/all_train_tokens_grams_{}_no_sw.p".format(df_param.iloc[i]['num_grams']), "rb"))

	val_data_tokens = pkl.load(open("/scratch/kyt237/NLP/HW1/pickle_no_sw/kyt237/val_data_tokens_grams_{}_no_sw.p".format(df_param.iloc[i]['num_grams']), "rb"))
	test_data_tokens = pkl.load(open("/scratch/kyt237/NLP/HW1/pickle_no_sw/kyt237/test_data_tokens_grams_{}_no_sw.p".format(df_param.iloc[i]['num_grams']), "rb"))

	# double checking
	print ("Train dataset size is {}".format(len(train_data_tokens)))
	print ("Val dataset size is {}".format(len(val_data_tokens)))
	print ("Test dataset size is {}".format(len(test_data_tokens)))
	print ("Total number of tokens in train dataset is {}".format(len(all_train_tokens)))

	train_target = pkl.load(open('/scratch/kyt237/NLP/HW1/pickles/train_target.p','rb'))
	val_target = pkl.load(open('/scratch/kyt237/NLP/HW1/pickles/val_target.p','rb'))
	test_target = pkl.load(open('/scratch/kyt237/NLP/HW1/pickles/test_target.p','rb'))

	VOCAB_SIZE = df_param.iloc[i]['vocab_sizes']
	token2id, id2token = build_vocab(all_train_tokens,VOCAB_SIZE)

	train_data_indices = token2index_dataset(train_data_tokens,token2id,id2token)
	val_data_indices = token2index_dataset(val_data_tokens,token2id,id2token)
	test_data_indices = token2index_dataset(test_data_tokens,token2id,id2token)

	# double checking
	print ("Train dataset size is {}".format(len(train_data_indices)))
	print ("Val dataset size is {}".format(len(val_data_indices)))
	print ("Test dataset size is {}".format(len(test_data_indices)))


	BATCH_SIZE = int(df_param.iloc[i]['batch_size'])
	#print(BATCH_SIZE)
	train_dataset = NewsGroupDataset(train_data_indices, train_target)
	train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
	                                           batch_size=BATCH_SIZE,
	                                           collate_fn=newsgroup_collate_func,
	                                           shuffle=True)

	val_dataset = NewsGroupDataset(val_data_indices, val_target)
	val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
	                                           batch_size=BATCH_SIZE,
	                                           collate_fn=newsgroup_collate_func,
	                                           shuffle=True)

	test_dataset = NewsGroupDataset(test_data_indices, test_target)
	test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
	                                           batch_size=BATCH_SIZE,
	                                           collate_fn=newsgroup_collate_func,
	                                           shuffle=False)

	emb_dim = df_param.iloc[i]['emb_dims']
	out_dim = 2
	model = BagOfWords(len(id2token), emb_dim,out_dim).to(DEVICE)


	learning_rate = df_param.iloc[i]['lrs']
	num_epochs = 20 # number epoch to train

	# Criterion and Optimizer
	criterion = torch.nn.CrossEntropyLoss()
	if(df_param.iloc[i]['optimizer'] == 'adam'):
	    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	else:
	    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)



	train_loss_history = []
	val_acc_history = []
	for epoch in range(num_epochs):
	    for j, (data, lengths, labels) in enumerate(train_loader):
	        model.train()
	        data_batch, length_batch, label_batch = data.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
	        optimizer.zero_grad()
	        outputs = model(data_batch, length_batch)
	        loss = criterion(outputs, label_batch)
	        loss.backward()
	        optimizer.step()
	        train_loss_history.append(loss.item())
	        # validate every 100 iterations
	        if j > 0 and j % 25 == 0:
	            # validate
	            #train_loss_history.append(loss.item())
	            print('Epoch: [{}/{}], Step: [{}/{}], Train Loss: {}'.format( 
	                       epoch+1, num_epochs, j+1, len(train_loader), loss.item()))

	    val_acc = test_model(val_loader, model)
	    val_acc_history.append(val_acc)
	    print('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}'.format( 
	                       epoch+1, num_epochs, j+1, len(train_loader), val_acc))



	val_acc_history = np.array(val_acc_history)
	max_val_acc = np.max(val_acc_history)
	max_val_acc_epoch = np.argmax(val_acc_history)

	df_param.at[i,'train_loss'] = train_loss_history
	df_param.at[i,'val_accs'] = val_acc_history
	df_param.at[i,'best_val_acc'] = max_val_acc
	df_param.at[i,'epoch_best_acc'] = max_val_acc_epoch+1
    
pkl.dump(df_param,open('result_df_no_sw.pkl','wb'))





