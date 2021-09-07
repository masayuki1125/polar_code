#!/usr/bin/env python
# coding: utf-8

# In[11]:


#必要なライブラリ、定数

from warnings import resetwarnings
import numpy as np
import math
from decimal import *
from AWGN import _AWGN

ch=_AWGN()


# In[12]:


class coding():
  def __init__(self,N):
    super().__init__()

    self.N=N
    self.R=0.5
    self.K=math.floor(self.R*self.N)
    self.design_SNR=4

    #for decoder
    self.n=0#calcurate codeword number
    self.k=0#calcurate information number
    self.EST_information=np.zeros(self.K)

    #prepere constants
    tmp2=np.log2(self.N)
    self.itr_num=tmp2.astype(int)
    self.frozen_bits,self.info_bits=self.Bhattacharyya_bounds()
    self.bit_reversal_sequence=self.reverse_bits()

    self.Gres=self.make_H() #no need

    self.filename="polar_code_{}_{}".format(self.N,self.K)

  #frozen_bitの選択
  def Bhattacharyya_bounds(self):
    E=np.zeros(1,dtype=np.float128)
    E =Decimal('10') ** (Decimal(str(self.design_SNR)) / Decimal('10'))
    
    z=np.zeros(self.N,dtype=np.float128)

    #10^10かけて計算する

    z[0]=math.exp(Decimal('-1')*Decimal(str(E)))

    #print("E=",np.exp(-E))

    for j in range(1,self.itr_num+1):
      tmp=2**(j)//2

      for t in range(tmp):
        T=z[t]
        z[t]=Decimal('2')*Decimal(str(T))-Decimal(str(T))**Decimal('2')
        z[tmp+t]=Decimal(str(T))**Decimal('2')
    #print(z)
    #np.savetxt("z",z)
    tmp=self.indices_of_elements(z,self.N)
    frozen_bits=tmp[:self.N-self.K]
    info_bits=tmp[self.N-self.K:]
    return np.sort(frozen_bits),np.sort(info_bits)

  @staticmethod
  def indices_of_elements(v,l):
    tmp=np.argsort(v)[::-1]
    #print(tmp)
    res=tmp[0:l]
    return res

  def reverse_bits(self):
      res=np.zeros(self.N,dtype=int)

      for i in range(self.N):
        tmp=format (i,'b')
        tmp=tmp.zfill(self.itr_num+1)[:0:-1]
        #print(tmp) 
        res[i]=int(tmp,2)
      
      return res
  
  #make parity check matrix

  @staticmethod
  def tensordot(A):
    tmp0=np.zeros((A.shape[0],A.shape[1]),dtype=np.int)
    tmp1=np.append(A,tmp0,axis=1)
    print(tmp1)
    tmp2=np.append(A,A,axis=1)
    print(tmp2)
    tmp3=np.append(tmp1,tmp2,axis=0)
    print(tmp3)
    return tmp3

  def make_H(self):
    G2=np.array([[1,0],[1,1]],dtype=np.int)
    Gres=G2
    for _ in range(self.itr_num-1):
      Gres=self.tensordot(Gres)
    return Gres


# In[15]:

class encoding(coding):
  def __init__(self,N):
    super().__init__(N)

  def generate_information(self):
    #generate information
    information=np.random.randint(0,2,self.K)
    return information

  def generate_U(self,information):
    u_message=np.zeros(self.N)
    u_message[self.info_bits]=information
    return u_message

  def encode(self,u_message):
    """
    Implements the polar transform on the given message in a recursive way (defined in Arikan's paper).
    :param u_message: An integer array of N bits which are to be transformed;
    :return: codedword -- result of the polar transform.
    """
    u_message = np.array(u_message)

    if len(u_message) == 1:
        codeword = u_message
    else:
        u1u2 = np.logical_xor(u_message[::2] , u_message[1::2])
        u2 = u_message[1::2]

        codeword = np.concatenate([self.encode(u1u2), self.encode(u2)])
    return codeword

  def polar_encode(self):
    information=self.generate_information()
    u_message=self.generate_U(information)
    #codeword=self.encode(u_message[self.bit_reversal_sequence])
    codeword=u_message@self.Gres%2
    return information,codeword

# In[19]:


#0,1が逆になって設計されているので、ちゃんと治す必要あり

class decoding(coding):

  def __init__(self,N):
    super().__init__(N)
    
  @staticmethod
  def chk(llr_1,llr_2):
    CHECK_NODE_TANH_THRES=30
    res=np.zeros(len(llr_1))
    for i in range(len(res)):

      if abs(llr_1[i]) > CHECK_NODE_TANH_THRES and abs(llr_2[i]) > CHECK_NODE_TANH_THRES:
        if llr_1[i] * llr_2[i] > 0:
          # If both LLRs are of one sign, we return the minimum of their absolute values.
          res[i]=min(abs(llr_1[i]), abs(llr_2[i]))
        else:
          # Otherwise, we return an opposite to the minimum of their absolute values.
          res[i]=-1 * min(abs(llr_1[i]), abs(llr_2[i]))
      else:
        res[i]= 2 * np.arctanh(np.tanh(llr_1[i] / 2, ) * np.tanh(llr_2[i] / 2))
    return res

  def SC_decoding(self,a):
    
    #interior node operation
    if a.shape[0]==1:

      #frozen_bit or not
      if np.any(self.frozen_bits==self.n):
        #print(decoding.check)
        tmp0=np.zeros(1)
      
      else :
        self.EST_information[self.k]=a
        self.k+=1
        #print(decoding.EST_information)

        if a>=0:
          tmp0=np.zeros(1)
        elif a<0:
          tmp0=np.ones(1)
      #print(decoding.n)
      #print(t)
      #decoding.n+=1
      #if t>=N:
        #exit()
      
      self.n+=1

      return tmp0

    #step1 left input a output u1_hat

    tmp1=np.split(a,2)
    f_half_a=self.chk(tmp1[0],tmp1[1])
    u1=self.SC_decoding(f_half_a)

    #step2 right input a,u1_hat output u2_hat 
    tmp2=np.split(a,2)
    g_half_a=tmp2[1]+(1-2*u1)*tmp2[0] 
    u2=self.SC_decoding(g_half_a)
  
    #step3 up input u1,u2 output a_hat
    res=np.concatenate([(u1+u2)%2,u2])
    return res
    

  def polar_decode(self,Lc):
    #initialize 
    self.n=0
    self.k=0

    self.SC_decoding(Lc)
    res=self.EST_information
    #err chenck
    if len(res)!=self.K:
      print("information length error")
      print(len(self.frozen_bits))
      print(self.n)
      print(len(res))
      exit()
    res=-1*np.sign(res)
    EST_information=(res+1)/2

    return EST_information


# In[29]:

class polar_code(encoding,decoding):
  def __init__(self,N):
    super().__init__(N)

  def main_func(self,EbNodB): 
    information,codeword=self.polar_encode()
    Lc=-1*ch.generate_LLR(codeword,EbNodB)#デコーダが＋、ー逆になってしまうので-１をかける
    EST_information=self.polar_decode(Lc)   
    if len(EST_information)!=len(information):
      print("len_err")
      exit()

    return information,EST_information

# In[30]:

#pc=polar_code(8)
#information,codeword=pc.polar_encode()
#Lc=-1*ch.generate_LLR(codeword,100)
#print(information)
#EST_information=pc.polar_decode(Lc)
#print(EST_information)

if __name__=="__main__":

  N=2048
  pc=polar_code(N)

  def output(EbNodB):
    count_err=0
    count_all=0
    count_berr=0
    count_ball=0
    MAX_ERR=8

    #seed値の設定
    np.random.seed()

    while count_err<MAX_ERR:
      
      information,EST_information=pc.main_func(EbNodB)
    
      if np.any(information!=EST_information):#BLOCK error check
        count_err+=1
      
      count_all+=1

      #calculate bit error rate 
      count_berr+=np.sum(information!=EST_information)
      count_ball+=len(information)

      print("\r","count_all=",count_all,",count_err=",count_err,"count_ball=",count_ball,"count_berr=",count_berr,end="")
      #import pdb; pdb.set_trace()

    print("BER=",count_berr/count_ball)
    return  count_err,count_all,count_berr,count_all
  
  output(4)
    
    


# In[ ]:

  #for i in range(-5,4):
      #print(i)

      #print(output(i))



