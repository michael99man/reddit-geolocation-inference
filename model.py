import numpy as np
import collections
import torch
import pickle
import time
import json
import torch.nn as nn
from  gensim.models import KeyedVectors
from featurization import word2vec, featurizer
from torch.utils.data import DataLoader
import geopy
import torch.optim as optim
from torch.utils.data import Dataset
# Set the device to use
# CUDA refers to the GPU
print("Let's use", torch.cuda.device_count(), "GPUs!")

## Hyperparameters
num_epochs = 100 
batch_size = 256
text_length = torch.ones([int(batch_size/8)])
print(text_length)

lstm_out_dim = 256
n_output_classes = 51
sub_dim = 5000
time_dim = 6

device = torch.device("cuda:0") 

def load_embedding():
    # load word2vec weights
    EpochSaver = word2vec.EpochSaver
    w2v = KeyedVectors.load_word2vec_format('./models/word2vec.model')

    weights = torch.FloatTensor(w2v.vectors)
    embedding = nn.Embedding.from_pretrained(weights)
    
    return embedding

def init_model():
    embedding = load_embedding()

    # instantiate the model
    model = Classifier(embedding)
    opt = optim.Adam(model.parameters())
    
    parallel_model = nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7])
    # move parallel model to device
    parallel_model=parallel_model.to(device)

    return train(parallel_model, opt, 0)

def continue_training(epoch):
    path = "epoch_" + str(epoch) +".model"
    checkpoint = torch.load(path)
    assert(checkpoint['epoch'] == epoch) 
   
    
    embedding = load_embedding()
    model = Classifier(embedding)
    
    parallel_model = nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7])
    # move parallel model to device
    parallel_model=parallel_model.to(device)
    parallel_model.load_state_dict(checkpoint['state_dict'])

    opt = optim.Adam(model.parameters())
    opt.load_state_dict(checkpoint['optimizer'])

    return train(parallel_model, opt, epoch+1)


def train(model, opt, epoch):
    # load saved feature tensors
    train_set = pickle.load(open("train_set.p", "rb"))
    
    # train_loader returns batches of training data. See how train_loader is used in the Trainer class later
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True,num_workers=30, drop_last = True)
    
    #opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    loss_function = nn.CrossEntropyLoss()

    trainer = Trainer(net=model, optim=opt, loss_function=loss_function, train_loader=train_loader, epoch=epoch)
    losses = trainer.train()
    
    torch.save(model.state_dict(), "final_model.pt")
    print("Saved models")
    
    return model

class ChungusSet(Dataset):
    def __init__(self, words, subs, times, labels):
        self.words = words
        self.subs = subs
        self.times = times
        self.labels = labels
        assert(len(words)== len(subs) and len(subs) == len(times) and len(times) == len(labels))
    
    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        #get images and labels here 
        #returned images must be tensor
        #labels should be int 
        return self.words[idx], self.subs[idx] , self.times[idx], self.labels[idx] 
        
class Classifier(nn.Module):
    def __init__(self, embedding):
        super(Classifier, self).__init__()

        self.embedding = embedding
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=lstm_out_dim,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.8)
        self.drop3 = nn.Dropout(p=0.3)
        # TODO: increase dimensions
        self.fc = nn.Linear(2*lstm_out_dim + sub_dim + time_dim, n_output_classes)

    def forward(self, words, subs, times, labels):
        x = self.embedding(words)
        x = self.drop(x)
        
        self.lstm.flatten_parameters()

        packed_embedded = nn.utils.rnn.pack_padded_sequence(x, text_length, batch_first=True)
        packed_output, (hidden, cell) = self.lstm(x)
        x = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        x = self.drop2(x)
       
        # apply aggressive dropout on subs
        subs = self.drop3(subs) 

        x = torch.cat((x, subs, times), dim=1)
        
        # TODO: concatenate with subreddit and metadata features
        x = self.fc(x)
        return x


### Drives training
class Trainer():
    def __init__(self,net=None,optim=None,loss_function=None, train_loader=None, epoch=0):
        self.net = net
        self.optim = optim
        self.loss_function = loss_function
        self.train_loader = train_loader
        self.epoch = epoch 

    def train(self):
        print("Beginning training at epoch %d" % self.epoch)
        losses = []
        
        while self.epoch < num_epochs:
            start = time.time()
            epoch_loss = 0.0
            epoch_steps = 0
            for (words, subs, times, labels) in self.train_loader:
                # ACT10-Zero the gradient in the optimizer, i.e. self.optim
                self.optim.zero_grad()
                
                # run the network on this input batch
                output = self.net(words, subs, times, labels)
                
                # move labels to output cuda
                labels = labels.to(output.device)
               
                loss = self.loss_function(output, labels.long())

                # ACT13-Backpropagate on the loss to compute gradients of parameters
                loss.backward()

                # ACT14-Step the optimizer, i.e. self.optim
                self.optim.step()
                epoch_loss += loss.item()
                epoch_steps += 1
            # average loss of epoch
            losses.append(epoch_loss / epoch_steps)
            print("epoch [%d]: loss %.3f, took %f seconds" % (self.epoch, losses[-1], time.time() - start))
            checkpoint = {
                    'epoch': self.epoch,
                    'state_dict': self.net.state_dict(),
                    'optimizer': self.optim.state_dict()
            }
            torch.save(checkpoint, "epoch_" + str(self.epoch) + ".model") 
            self.epoch +=1 
        return losses



def evaluate_model(model):
    test_set = pickle.load(open("test_set.p", "rb"))
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True,num_workers=30, drop_last=True)

    err=0
    tot = 0
    with torch.no_grad():
        for (words, subs, times, labels) in test_loader:
            output = model(words, subs, times, labels)

            # let the maximum index be our predicted class
            _, yh = torch.max(output, 1)

            tot += labels.size(0)

            ## add to err number of missclassification, i.e. number of indices that
            ## yh and y are not equal
            ## note that y and yh are vectors of size = batch_size = (256 in our case)
            err += sum(list(map(lambda i: 1 if labels[i] != yh[i] else 0, range(len(labels)))))

    print('Accuracy of FC prediction on test digits: %5.2f%%' % (100-100 * err / tot))

if __name__ == "__main__":
    completed_model = continue_training(42)
    evaluate_model(completed_model)

