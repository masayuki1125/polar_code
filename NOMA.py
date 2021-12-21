#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
from polar_code import polar_code
from AWGN import _AWGN


# In[23]:


#最適なβの値の設計方法
#①N1=N2にし、固定する
#②Strong Userの受信SNRを決め、固定する
#③シャノン限界を基準にして、Strong UserのRateが0.8になるようにβ1、β2を設計
#④β=β1/β2とし、Weak Userの受信SNRを変化させて、全体のsystemのBERを測定する


# In[59]:


class NOMA():
  def __init__(self,N):
    #EbNodB1>EbNodB2
    #User1=Strong User(Fixed)
    #User2=Weak User
    self.EbNodB1=10
    
    #Strong Userの受信SNから、βを決定する
    self.beta=0.1
    
    #self.EbNodB2 change
    
    EbNo1 = 10 ** (self.EbNodB1 / 10)
    self.No1=1/EbNo1
    
    self.ch=_AWGN()
    self.cd1=polar_code(N//2)
    self.cd2=polar_code(N//2)
    
    self.filename="NOMA_polar_{}_{}".format(N,N//2)
    
  def NOMA_encode(self):
    info1,cwd1=self.cd1.polar_encode()
    info2,cwd2=self.cd2.polar_encode()
    return info1,info2,cwd1,cwd2


# In[60]:


'''
class NOMA(NOMA):
  def make_beta(EsNodB):
    EsNo = 10 ** (EsNodB / 10)
    No=1/EsNo
    #Strong UserのCapacityを求める
    x, y = sym.symbols("x y")
    st_usr=sym.log(1+x/No,2)
'''


# In[61]:


class NOMA(NOMA):
  def channel(self,cwd1,cwd2,beta):
    
    const1=self.ch.generate_QAM(cwd1)
    const2=self.ch.generate_QAM(cwd2)
    res_const=beta*const1+const2
    
    return res_const
  
  def decode1(self,res_const,No1):
    '''
    decode using SIC
    input generate_constellation,Noise variance
    output estimated information
    '''
    
    EST_info2=self.decode2(res_const,No1+self.beta)
    
    #re encode polar code
    u_message=self.cd2.generate_U(EST_info2)
    EST_cwd2=self.cd2.encode(u_message[self.cd2.bit_reversal_sequence])
    
    EST_const2=self.ch.generate_QAM(EST_cwd2)

    RX_const=res_const-EST_const2

    Lc=-1*self.ch.demodulate(RX_const,No1/self.beta)
    EST_info1=self.cd1.polar_decode(Lc)
    
    return EST_info1
  
  def decode2(self,res_const,No2):
    RX_const=self.ch.add_AWGN(res_const,No2+self.beta)
    Lc=-1*self.ch.demodulate(RX_const,No2+self.beta)
    EST_info2=self.cd2.polar_decode(Lc)
    
    return EST_info2
  
  def NOMA_decode(self,res_const,No1,No2):
    EST_info1=self.decode1(res_const,No1)
    EST_info2=self.decode2(res_const,No2)
    
    return EST_info1,EST_info2
  
  def main_func(self,EbNodB2):
    #make No2
    EbNo2 = 10 ** (EbNodB2 / 10)
    No2=1/EbNo2
    
    info1,info2,cwd1,cwd2=self.NOMA_encode()
    res_const=self.channel(cwd1,cwd2,self.beta)
    EST_info1,EST_info2=self.NOMA_decode(res_const,self.No1,No2)
    
    info=np.concatenate([info1,info2])
    #cwd=np.concatenate([cwd1,cwd2])
    EST_info=np.concatenate([EST_info1,EST_info2])
    
    return info,EST_info
    


# In[68]:


if __name__=="__main__":
  ma=NOMA(1024)
  a,b=ma.main_func(-1)
  print(np.sum(a!=b))

