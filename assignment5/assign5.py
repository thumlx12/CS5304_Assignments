
# coding: utf-8

# In[1]:


import numpy as np
from scipy.sparse import rand as sprand
from scipy.sparse import lil_matrix
import torch
from torch.autograd import Variable
import pandas as pd
import copy


# In[2]:


EPOCH = 50
BATCH_SIZE = 1000
LR = 3e-3
n_factor = 4
use_gpu = torch.cuda.is_available()


# In[9]:


trains = ['ml-10M100K/r1.train','ml-10M100K/r2.train','ml-10M100K/r3.train','ml-10M100K/r4.train','ml-10M100K/r5.train']
tests = ['ml-10M100K/r1.test','ml-10M100K/r2.test','ml-10M100K/r3.test','ml-10M100K/r4.test','ml-10M100K/r5.test']


# In[10]:


names = ['user_id', 'item_id', 'rating', 'timestamp']
df_trains =[pd.read_csv(t, sep='::', names=names,engine='python') for t in trains]
df_tests = [pd.read_csv(t, sep='::', names=names,engine='python') for t in tests]


# In[11]:


def get_movielens_ratings(df):
    n_users = max(df.user_id.unique())
    n_items = max(df.item_id.unique())

    interactions = lil_matrix( (n_users,n_items), dtype=float) #np.zeros((n_users, n_items))
    for row in df.itertuples():
        interactions[row[1] - 1, row[2] - 1] = row[3]
    return interactions


# In[16]:


ratings_arr = [get_movielens_ratings(df) for df in df_trains]
ratings_arr_test = [get_movielens_ratings(df) for df in df_tests]


# In[24]:


def get_batch(batch_size,ratings):
    # Sort our data and scramble it
    rows, cols = ratings.shape
    p = np.random.permutation(rows)
    
    # create batches
    sindex = 0
    eindex = batch_size
    while eindex < rows:
        batch = p[sindex:eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch
    
    if eindex >= rows:
        batch = range(sindex,rows)
        yield batch
        
def get_batch_in_seqence(batch_size, ratings):
    rows, cols = ratings.shape
    # create batches
    sindex = 0
    eindex = batch_size
    while eindex < rows:
        batch = range(sindex,eindex)
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch
    if eindex >= rows:
        batch = range(sindex,rows)
        yield batch


# In[25]:


def run_epoch(model, batch_size, ratings, loss_func, reg_loss_func):
    epoch_ave_loss = 0.0
    batch_cnt = 0
    for i,batch in enumerate(get_batch(batch_size, ratings)):
        # Set gradients to zero
        reg_loss_func.zero_grad()
        
        # Turn data into variables
        interactions = Variable(torch.FloatTensor(ratings[batch, :].toarray()))
        rows = Variable(torch.LongTensor(batch))
        cols = Variable(torch.LongTensor(np.arange(ratings.shape[1])))
        if use_gpu:
            interactions = interactions.cuda()
            rows = rows.cuda()
            cols = cols.cuda()
    
        # Predict and calculate loss
        predictions = model(rows, cols)
        loss = loss_func(predictions, interactions)
        epoch_ave_loss += loss.data[0]
        batch_cnt += 1
        # Backpropagate
        loss.backward()
    
        # Update the parameters
        reg_loss_func.step()
        
    return model, epoch_ave_loss/batch_cnt

def run_test_batchly(model, ratings, loss_func, batch_size=1000):
    for i,batch in enumerate(get_batch_in_seqence(batch_size, ratings)):
        # Turn data into variables
        interactions = Variable(torch.FloatTensor(ratings[batch, :].toarray()))
        rows = Variable(torch.LongTensor(batch))
        cols = Variable(torch.LongTensor(np.arange(ratings.shape[1])))
        if use_gpu:
            interactions = interactions.cuda()
            rows = rows.cuda()
            cols = cols.cuda()
        # Predict and calculate loss
        predictions = model(rows, cols)
        loss = loss_func(predictions, interactions)
        yield predictions, loss.data[0]


# In[26]:


def train_model(model, ratings, weight_decay, num_epochs=25, batch_size=1000):
    reg_loss_func = torch.optim.SGD(model.parameters(), lr = 3e-3, weight_decay = weight_decay)
    loss_func = torch.nn.MSELoss()
    best_model_wts = copy.deepcopy(model.state_dict())
    
    lowest_loss = 1e10
    print('Epoch\tAve-loss')
    for epoch in range(num_epochs):
        model, ave_loss = run_epoch(model, batch_size, ratings, loss_func, reg_loss_func)
        print('{}\t{}'.format(epoch, ave_loss))
        if lowest_loss > ave_loss:
            lowest_loss = ave_loss
            best_model_wts = copy.deepcopy(model.state_dict())
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# In[18]:


class MatrixFactorization(torch.nn.Module):    
    def __init__(self, n_users, n_items, n_factors=4):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors, sparse=False)
        self.item_factors = torch.nn.Embedding(n_items, n_factors, sparse=False)
        if use_gpu:
            self.user_factors = self.user_factors.cuda()
            self.item_factors = self.item_factors.cuda()
        # Also should consider fitting overall bias (self.mu term) and both user and item bias vectors
        # Mu is 1x1, user_bias is 1xn_users. item_bias is 1xn_items
    
    # For convenience when we want to predict a sinble user-item pair. 
    def predict(self, user, item):
        # Need to fit bias factors
        return (pred + self.user_factors(user) * self.item_factors(item)).sum(1)
    
    # Much more efficient batch operator. This should be used for training purposes
    def forward(self, users, items):
        # Need to fit bias factors
        return torch.mm(self.user_factors(users),torch.transpose(self.item_factors(items),0,1))


# In[20]:


class BiasedMatrixFactorization(torch.nn.Module):
    
    def __init__(self, n_users, n_items, n_factors=4):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors, sparse=False)
        self.item_factors = torch.nn.Embedding(n_items, n_factors, sparse=False)
        self.item_biases = torch.nn.Embedding(n_items, 1, sparse=False)
        if use_gpu:
            self.user_factors = self.user_factors.cuda()
            self.item_factors = self.item_factors.cuda()
            self.item_biases = self.item_biases.cuda()
        
    def forward(self, users, items):
        constant_user_biases = Variable(torch.FloatTensor(np.transpose([np.ones(len(users))])))
        if use_gpu:
            constant_user_biases = constant_user_biases.cuda()
        biases = torch.mm(constant_user_biases, torch.transpose(self.item_biases(items),0,1))
        linear = torch.mm(self.user_factors(users),torch.transpose(self.item_factors(items),0,1))
        return biases + linear


# In[15]:


models_without_biases = {}
for i in range(5):
    ratings = ratings_arr[i]
    user_num, item_num = ratings.shape
    for weight_decay in [0.001, 0.01, 0.1]:
        print('trainset:{}, weight_decay:{}'.format(i, weight_decay))
        model = MatrixFactorization(user_num, item_num, 4)
        if use_gpu:
            model = model.cuda()
        model = train_model(model, ratings, weight_decay, EPOCH)
        model_name = str(i)+ '_' + str(weight_decay)
        models_without_biases[model_name] = model


# In[ ]:


models_with_biases = {}
for i in range(5):
    ratings = ratings_arr[i]
    user_num, item_num = ratings.shape
    for weight_decay in [0.1, 0.01, 0.001]:
        print('trainset:{}, weight_decay:{}'.format(i, weight_decay)) 
        model = BiasedMatrixFactorization(user_num, item_num, 4)
        if use_gpu:
            model = model.cuda()
        model = train_model(model, ratings, weight_decay, EPOCH)
        model_name = str(i)+ '_' + str(weight_decay)
        models_with_biases[model_name] = model


# In[69]:


for item in models_with_biases.items():
    torch.save(item[1],'model/with_biases_'+str(item[0]))
    
for item in models_without_biases.items():
    torch.save(item[1],'model/without_biases_'+str(item[0]))


# In[22]:


bias_model = models_with_biases['0_0.1']
rst = open('assign5_r5results.tsv','w')
u_ids_in_test = set(df_tests[4].user_id.unique())
user_id = 1
for pred, loss in run_test_batchly(bias_model, ratings_arr_test[4], torch.nn.MSELoss()):
    for user in pred:
        if user_id %10000 == 0: 
            print(user_id)
        tops = torch.topk(user,50)[1].data.cpu().numpy()
        recs = np.setdiff1d(tops, np.array(ratings_arr_test[4].rows[user_id-1]), assume_unique=True)
        if user_id in u_ids_in_test:
            recs = recs[:5]
            rst.write(str(user_id) + '\t' + '\t'.join([str(rec) for rec in recs]) + '\n')
        user_id += 1
rst.close()      

