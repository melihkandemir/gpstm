#
# The Gaussian Process Supervised Topic Model (GPSTM) code
#
# Copyright: Melih Kandemir
# melih.kandemir@iwr.uni-heidelberg.de
#
# All rights reserved
#
import numpy as np
import scipy
from RBFKernel import RBFKernel
import sys
from copy import deepcopy
from sklearn.cluster import KMeans

class GPSTM:

    # K: topic count, V: vocabulary size
    def __init__(self,num_inducing=20,K=20,max_iter=10,burnin_iter=5,length_scale=1.0,noise_precision=10.0,lrate=0.01):
        self.P = num_inducing;   self.kernel = RBFKernel(length_scale)
        self.K = K;  self.V = 1;   self.max_iter=max_iter;   
        self.burnin_iter=burnin_iter; self.lrate=lrate
        self.noise_precision= noise_precision;   self.num_inducing = num_inducing
        
    # Xtr: training input (data points in rows)
    # Ytr: training output (labels in rows)
    def train(self,Xtr,Ytr):
        Xtr = np.float32(Xtr);   Ytr=np.float32(Ytr);   P=self.num_inducing
        D = Xtr.shape[0];   K = self.K;   V = Xtr.shape[1]; R = Ytr.shape[1] # num: classes
        InpNoise=np.ones([2,2])*1e-7;   word_cnt = np.sum(Xtr,axis=1); word_cnt[word_cnt==0.0]=1.0;   C=-1.0
        
        # initialize       
        # topic distributions
        gam = np.random.random([D,K])   
        gam = gam / np.tile(np.sum(gam,axis=1),[K,1]).T          
        # topic assignments
        phi = np.random.random([D,K,V])
        phi = phi / np.swapaxes(np.tile(np.sum(phi,axis=1),[K,1,1]),0,1)
        # hyperparams        
        alpha = np.ones([K,1]).ravel()
        beta =  np.random.random([K,V])
        beta = beta / np.tile(np.sum(beta,axis=1),[V,1]).T
        # gradient of C
        doc_means = np.zeros([D,K])
        C = np.zeros([D,K])
        gC = np.zeros([D,K])

        print "Training"

        # Train LDA
        for ii in range(self.max_iter):  
            self.print_iter(ii)   
            (beta,gam,phi)=self.lda_update(alpha,beta,gam,phi,Xtr,C,joint_update=(ii>self.burnin_iter))
            
            if ii==self.burnin_iter: # Compute inducing points only once!                        
                for dd in range(D):
                    doc_means[dd,:] = phi[dd,:,:].dot(Xtr[dd,:])/word_cnt[dd]                
                C = deepcopy(doc_means)           
                kmmodel=KMeans(n_clusters=P, n_init=1,init='random')
                
                for rr in range(R):                    
                    kmmodel.fit(np.float32(C[Ytr[:,rr]==1,:]))                    
                    Z_rr = kmmodel.cluster_centers_
                    if rr == 0:
                        Z = Z_rr
                    else:
                        Z = np.concatenate((Z,Z_rr))   
                # calculate kernels
                Kzz = self.kernel.selfCompute(Z)
                Kzz_inv = self.safe_inv(Kzz)                
                
                (M,S,EKzc,EKzcKzcT)=self.gp_update(Kzz_inv,Z,C,Ytr,InpNoise)            
            
            if ii > self.burnin_iter:                            
               for dd in range(D):
                   doc_means[dd,:] = phi[dd,:,:].dot(Xtr[dd,:])/word_cnt[dd]  
                       
               if np.mod(ii-self.burnin_iter,2)==0:                     
                   C = deepcopy(doc_means)                      
               else:
                                                
                    # update GP params                            
                    if ii>self.burnin_iter:
                        lrnow=self.lrate
                        for ppp in range(10):
                            gC = -self.noise_precision*C + self.noise_precision*doc_means
                        
                            for kk in range(K):
                                grad_EKcz = self.kernel.grad_EVzx_by_mu_batch(EKzc, Z, C, InpNoise, kk)      
                                grad_EKzczcT_tensor = self.kernel.grad_EVzxVzxT_by_mu_batch(EKzcKzcT, Z, C, InpNoise, kk)
                                Multiplier = Kzz_inv.dot(M.dot(M.T)+R*S).dot(Kzz_inv)-R*Kzz_inv
                                Term2 = np.zeros([D,])
                                for dd in range(D):
                                    Term2[dd] = grad_EKzczcT_tensor[dd,:,:].dot(Multiplier).trace()
                                
                                gC[:,kk] += -0.5*np.float64(self.noise_precision)*Term2
                                
                                #print kk
                                for rr in range(R):                                            
                            
                                    label_mat = np.tile(Ytr[:,rr],[P*R,1]).T                                                                
                                    gC[:,kk]  +=  np.float64(self.noise_precision) * (label_mat*grad_EKcz).dot( Kzz_inv).dot(M[:,rr]).ravel()
                           
                            C = C + lrnow*gC
                            lrnow *= 0.9

		    (M,S,EKzc,EKzcKzcT)=self.gp_update(Kzz_inv,Z,C,Ytr,InpNoise)
                
        # save the learned params
        self.M = M;  self.S = S;  self.Z = Z;  self.Kzz_inv = Kzz_inv
        self.gam = gam;  self.beta = beta;  self.alpha = alpha

    def lda_update(self,alpha,beta,gam,phi,X,C,update_beta=True,joint_update=False):  
        X = np.float32(X)
        D = X.shape[0];  K = gam.shape[1];   V = X.shape[1]
        word_cnt = np.sum(X,axis=1); word_cnt[word_cnt==0.0]=1.0
        beta_new = np.zeros([K,V]) if update_beta else 0    
        
        for dd in range(D):
            expPsiGam = np.exp(scipy.special.psi(gam[dd,:]))

            if joint_update: # Consume more memory only if needed
                phi_dd_old = deepcopy(phi[dd,:,:])

            phi[dd,:,:] = np.tile(expPsiGam,[V,1]).T * beta
            if joint_update:
                sum_phis_but_one = np.tile(phi[dd,:,:].dot(X[dd,:]),[V,1]).T - phi_dd_old                
                phi[dd,:,:] *= np.exp(self.noise_precision/word_cnt[dd]*np.tile(C[dd,:],[V,1]).T-0.5*self.noise_precision/(word_cnt[dd]**2.0)*(sum_phis_but_one+1.0))
                
            phi_dd_sum= np.tile(np.sum(phi[dd,:],axis=0),[K,1])
            phi_dd_sum[phi_dd_sum==0] = 1.0
            phi[dd,:,:] = phi[dd,:,:] / phi_dd_sum
            
            gam[dd,:] = alpha + phi[dd,:].dot(X[dd,:])
            if update_beta:             
                beta_new += phi[dd,:] * np.tile(X[dd,:],[K,1])
        
        # update beta
        if update_beta:
            beta = beta_new / np.tile(np.sum(beta_new,axis=1),[V,1]).T        
        return (beta,gam,phi)

    def gp_update(self,Kzz_inv,Z,C,Ytr,InpNoise):

        EKzc = self.kernel.EVzx(Z,C,InpNoise)
        EKzcKzcT = np.sum(self.kernel.EVzxVzxT(Z,C,InpNoise),axis=0)
            
        S = self.safe_inv(Kzz_inv + self.noise_precision*Kzz_inv.dot(EKzcKzcT).dot(Kzz_inv))
        M = self.noise_precision*S.dot(Kzz_inv).dot(EKzc).dot(Ytr)

        return (M,S,EKzc,EKzcKzcT)
        
    def print_iter(self,ii):
        if (ii>0 and np.mod(ii,10)==0) or ii==(self.max_iter-1):
            print ". %d" % ii
        elif ii>0:
            print ".", 
            sys.stdout.flush()        
           
    def safe_inv(self,M):
        return np.linalg.inv(M + np.identity(M.shape[0])*0.0001)
        
    def predict(self,Xts):
        Xts = np.float32(Xts);   word_cnt = np.sum(Xts,axis=1);  word_cnt[word_cnt==0.0]=1.0       
        D = Xts.shape[0];   K=self.beta.shape[0];   V = Xts.shape[1]
        gam = np.random.random([D,K])   
        gam = gam / np.tile(np.sum(gam,axis=1),[K,1]).T
        phi = np.random.random([D,K,V])
        phi = phi / np.swapaxes(np.tile(np.sum(phi,axis=1),[K,1,1]),0,1)
        
        print "Predicting"
        
        for ii in range(self.max_iter):  
            self.print_iter(ii)   
            (dummy,gam,phi)=self.lda_update(self.alpha,self.beta,gam,phi,Xts,\
                                            None,update_beta=False,joint_update=False)
        Cts = np.zeros([D,K])
        for dd in range(D):
            Cts[dd,:]= phi[dd,:].dot(Xts[dd,:])/word_cnt[dd]

        Kts = self.kernel.compute(Cts,self.Z)  
        return  Kts.dot(self.Kzz_inv).dot(self.M)
