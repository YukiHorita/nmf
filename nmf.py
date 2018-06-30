import pandas as pd
import numpy as np

class NMF(object):
    def __init__(self,N,k):
        self.N=np.array(N)
        self.P=self.N.shape[0]#行の数
        self.Q=self.N.shape[1]#列の数
        self.K=k#トピック数
        self.pre_llh=0.1
        #################################
        #トピック数
        #初期化
        self.H=np.random.rand(self.P,k)
        self.U=np.random.rand(k,self.Q)
        #################################

    def step(self):
        pre_H=self.H
        pre_U=self.U

        self.H=pre_H*(self.N@self.U.T)/(self.H@self.U@self.U.T)
        self.U=pre_U*(self.H.T@self.N)/(self.H.T@self.H@self.U)

    def llh(self,t=0.05):
        zansa=np.sum(((self.N-self.H@self.U)**2)**2)
        return zansa

    def train(self,i=10000000,k=1.00e-6):
        for x in range(i):
            self.step()
            zansa=self.llh()
            if np.abs(zansa-self.pre_llh)/zansa<k:
                break
            self.pre_llh=zansa

if __name__=="__main__":
    dn=np.random.rand(100,100)*100
    nmf=NMF(dn,20)
    nmf.train()
    print(nmf.H)
    print(nmf.U)
    pd.DataFrame(nmf.H).to_csv("H.csv",header=None,index=None)
    pd.DataFrame(nmf.U).to_csv("U.csv",header=None,index=None)
