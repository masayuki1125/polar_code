#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from polar_code import polar_code
from AWGN import _AWGN


# In[3]:


#最適なβの値の設計方法
#①N1=N2にし、固定する
#②Strong Userの受信SNRを決め、固定する
#③シャノン限界を基準にして、Strong UserのRateが0.8になるようにβ1、β2を設計
#④β=β1/β2とし、Weak Userの受信SNRを変化させて、全体のsystemのBERを測定する


# In[3]:


class NOMA():
  def __init__(self,N,beta1=0.2):
    self.N=N
    self.K=self.N//2
    
    self.N1=self.N//2
    self.K1=self.K//2
    self.N2=self.N//2
    self.K2=self.K//2
    #EbNodB1>EbNodB2
    #User1=Strong User(Fixed)
    #User2=Weak User
    self.EbNodB_diff=10
    
    #Strong Userの受信SNから、βを決定する
    print("beta1")
    print(beta1)
    self.beta=(beta1**(1/2))/((1-beta1)**(1/2))
    print(self.beta)
    print("beta")
    
    
    #self.EbNodB2 change
    
    #EbNo1 = 10 ** (self.EbNodB1 / 10)
    #self.No1=1/EbNo1
    
    self.ch=_AWGN()
    self.cd1=polar_code(self.N1,self.K1)
    self.cd2=polar_code(self.N2,self.K2)
    
    self.filename="NOMA_polar_{}_{}_{}".format(self.beta,self.N,self.K)
    
  def NOMA_encode(self):
    info1,cwd1=self.cd1.polar_encode()
    info2,cwd2=self.cd2.polar_encode()
    return info1,info2,cwd1,cwd2


# In[29]:


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
    EbNodB1=EbNodB2+self.EbNodB_diff
    EbNo1 = 10 ** (EbNodB1 / 10)
    No1=1/EbNo1
    
    EbNo2 = 10 ** (EbNodB2 / 10)
    No2=1/EbNo2
    
    #change construction
    if EbNodB1!=self.cd1.design_SNR:
      self.cd1.design_SNR=EbNodB1
      self.cd1.frozen_bits,self.cd1.info_bits=self.cd1.const.main_const(self.N1,self.K1,self.cd1.design_SNR)#,self.beta)
      
    if EbNodB2!=self.cd2.design_SNR:
      self.cd2.design_SNR=EbNodB2
      self.cd2.frozen_bits,self.cd2.info_bits=self.cd2.const.main_const(self.N2,self.K2,self.cd2.design_SNR)
    
    info1,info2,cwd1,cwd2=self.NOMA_encode()
    res_const=self.channel(cwd1,cwd2,self.beta)
    EST_info1,EST_info2=self.NOMA_decode(res_const,No1,No2)
    
    info=np.concatenate([info1,info2])
    #cwd=np.concatenate([cwd1,cwd2])
    EST_info=np.concatenate([EST_info1,EST_info2])
    
    return info,EST_info
    


# In[31]:


if __name__=="__main__":
  ma=NOMA(1024)
  a,b=ma.main_func(100)
  print(np.sum(a!=b))
  print(a!=b)


# In[ ]:




