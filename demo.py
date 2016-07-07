#
# Sample binary text categorization application on TechTC-100 data
#
# Copyright: Melih Kandemir
# melih.kandemir@iwr.uni-heidelberg.de
#
# All rights reserved
#
import numpy as np
from GPSTM import GPSTM
import scipy.io
import scipy
import scipy.special
        
np.random.seed(483)

mat = scipy.io.loadmat( 'data/tech1.mat' )
Xtr = mat["Xtr"]
ytr = mat["ytr"]
Xts = mat["Xts"]
yts = mat["yts"]

#techtc (-1,1) -> (0,1)    
repl = np.where(ytr == -1); ytr[repl] = 0;
repl = np.where(yts == -1); yts[repl] = 0;

#
used_words=np.where(np.sum(Xtr,axis=0)>0)[0]
Xtr=Xtr[:,used_words]
Xts=Xts[:,used_words]
word_freq_order=np.argsort(np.sum(Xtr,axis=0))
Xtr=Xtr[:,word_freq_order[-1000:]]
Xts=Xts[:,word_freq_order[-1000:]]

#throw out documents having zero word sum
nzd = np.where(np.sum(Xtr,axis=1) > 0)[0]
Xtr = Xtr[nzd,:]
ytr=ytr[nzd]
nzd = np.where(np.sum(Xts,axis=1) > 0)[0]
Xts = Xts[nzd,:]
yts=yts[nzd] 

# Setup
isjoint=True   # True: our target model, False: the disjoint LDA+GP baseline
ind=20         # num inducing points
num_top=5;     # num topics
sig=0.5;        # kernel bandwidth
itr_cnt=50;     # iteration count
burnin_cnt=40;  # burn-in iteration count (LDA only)
lrate=1e-7      # learning rate (1e-7 is not bad)

# Calculate base rate (percentage of the majority class)
naiveBaseAcc = (1-np.mean(yts.ravel()>0.0))*100.0;

if isjoint==False:
    burnin_cnt=itr_cnt-1 # comment in for baseline 1 (what happens if we keep iterating disjointly)
print ("Setup: isjoint:%d, ind=%d, K=%d, sigma=%.2f, max_iter: %d, burnin_iter: %d" % (isjoint,ind,num_top,sig,itr_cnt,burnin_cnt))

## Our model
model = GPSTM(num_inducing=ind,K=num_top,length_scale=sig,max_iter=itr_cnt,burnin_iter=burnin_cnt,lrate=lrate)
model.train(Xtr,ytr)
ypred = model.predict(Xts)

naiveBaseAcc = np.mean(yts.ravel()>0.0)*100.0;
acc = np.mean(yts.ravel()==(ypred.ravel()>0.5))*100.0;
print ("Test Accuracy: %.1f Base: %.1f" % (acc, naiveBaseAcc) )
print(isjoint)