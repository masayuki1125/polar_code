#!/usr/bin/env python
# coding: utf-8

# In[32]:


#必要なライブラリ、定数
import sys
sys.path.append("../channel")
from AWGN import _AWGN

import numpy as np
import random
import time
import math
from decimal import *
random.seed(time.time())

ch=_AWGN()


# In[44]:


class coding():
    def __init__(self,N):
        super().__init__()

        self.N=N
        self.R=0.5
        self.K=math.floor(self.R*self.N)
        self.design_SNR=4

        #prepere constants
        tmp2=np.log2(self.N)
        tmp2=tmp2.astype(int)
        self.itr_num=tmp2
        self.frozen_bits,self.info_bits=            Bhattacharyya_bounds(self.N,self.K,self.design_SNR)

        self.Gres=make_H(self.itr_num)

        self.filename="polar_code_{}_{}".format(self.N,self.K)
        


# In[34]:


#cd=coding(1024)
#a=512.0
#a=a.astype(int)
#print(a)


# In[35]:


#frozen_bitの選択
def Bhattacharyya_bounds(N,K,EdB):
  E=np.zeros(1,dtype=np.float128)
  E =Decimal('10') ** (Decimal(str(EdB)) / Decimal('10'))
  itr_num=np.log2(N)
  itr_num=itr_num.astype(int)
  z=np.zeros(N,dtype=np.float128)

  #10^10かけて計算する

  z[0]=math.exp(Decimal('-1')*Decimal(str(E)))

  #print("E=",np.exp(-E))

  for j in range(1,itr_num+1):
    tmp=2**(j)//2

    for t in range(tmp):
      T=z[t]
      z[t]=Decimal('2')*Decimal(str(T))-Decimal(str(T))**Decimal('2')
      z[tmp+t]=Decimal(str(T))**Decimal('2')
  #print(z)
  #np.savetxt("z",z)
  tmp=indices_of_elements(z,N)
  frozen_bits=tmp[:N-K]
  info_bits=tmp[N-K:]
  return np.sort(frozen_bits),np.sort(info_bits)

def indices_of_elements(v,l):
  tmp=np.argsort(v)[::-1]
  #print(tmp)
  res=tmp[0:l]
  return res


# In[36]:


def generate_information(K):
      #generate information
  information=np.random.randint(0,2,K)
  return information


# In[37]:


def generate_U(N,information,info_bits):
    u_message=np.zeros(N)
    u_message[info_bits]=information
    return u_message


# In[38]:


def encode(u_message):
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

        codeword = np.concatenate([encode(u1u2), encode(u2)])
    return codeword


# In[39]:


def tensordot(A):

  tmp0=np.zeros((A.shape[0],A.shape[1]),dtype=np.int)
  tmp1=np.append(A,tmp0,axis=1)
  #print(tmp1)
  tmp2=np.append(A,A,axis=1)
  #print(tmp2)
  tmp3=np.append(tmp1,tmp2,axis=0)
  #print(tmp3)
  return tmp3

def make_H(itr_num):
  G2=np.array([[1,0],[1,1]],dtype=np.int)
  Gres=G2
  for i in range(itr_num-1):
    #print(i)
    Gres=tensordot(Gres)
  return Gres

'''
N=16
K=8
itr_num=np.log2(N)
itr_num=itr_num.astype(int)

Gres=make_H(itr_num)
#print(Gres)

cd=coding()
info=generate_information(K)
print(info)
inbits=np.sort(cd.info_bits)
print(inbits)

u_message=generate_U(N,info,inbits)
print(u_message)
S1=(u_message@Gres)%2
print(S1)
S2=encode(u_message)
print(S2)

Lc=-1*ch.generate_LLR(S2,100)
print(Lc)

dc=decoding()
dc.SC_decoding(Lc)
res=dc.EST_information
print(res)
res=-1*np.sign(res)
codeword=(res+1)/2
print(codeword)

#codeword=codeword[inbits]
#print(codeword)
codeword=codeword[inbits]
print(codeword)
'''


# In[40]:


#0,1が逆になって設計されているので、ちゃんと治す必要あり

class decoding(coding):
  n=0
  EST_information=np.array([])

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
      if np.any(self.frozen_bits==decoding.n):
        tmp0=np.zeros(1)
      elif a>=0:
        tmp0=np.zeros(1)
      elif a<0:
        tmp0=np.ones(1)
      else:
        print("err!")
        exit()
      
      if np.any(self.info_bits==decoding.n):
        decoding.EST_information=np.append(decoding.EST_information,a)
      #print(decoding.n)
      #print(t)
      decoding.n+=1
      #if t>=N:
        #exit()
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


# In[41]:


'''
import numpy as np

class parent():
    def __init__(self):
        self.a=10

class child(parent):
    @classmethod
    def set_v(cls):
        child.i=0
        b=np.zeros(cls.a)

    def func(self):
        if child.ser_v.i<5:
            child.ser_v.i+=1
            print(child.ser_v.i)
            print("p.a=",self.a)
            child.b[child.ser_v.i]=child.ser_v.i
            print(child.ser_v.b)
            c=child()
            c.func()


c=child()
c.func()
print("i",child.i)
print("b",child.b)
'''


# In[42]:


class polar_code(decoding):
    def __init__(self,N):
        super().__init__(N)

    def encode(self):
        information=generate_information(self.K)
        #information=np.array([1,0,0,1])
        u_message=generate_U(self.N,information,self.info_bits)
        codeword=(u_message@self.Gres)%2
        #codeword=encode(u_message)
        return information,codeword

    def decode(self,Lc):
        self.SC_decoding(Lc)
        res=self.EST_information #デコーダが＋、ー逆になってしまうので-１をかける
        res=-1*np.sign(res)
        res=(res+1)/2
        return res

    def polar_code(self,EbNodB):
        information,codeword=self.encode()
        Lc=-1*ch.generate_LLR(codeword,EbNodB)#デコーダが＋、ー逆になってしまうので-１をかける
        EST_information=self.decode(Lc)
        return information,EST_information


# In[43]:


if __name__=="__main__":
    pc=polar_code(512)
    print(len(pc.info_bits))

    information,EST_information=pc.polar_code(100)
    print(len(information))
    print(len(EST_information))
    print(np.sum(information!=EST_information))
    


# In[ ]:


'''
EbNodB_range=np.arange(0,5.5,0.5)
BLER=np.zeros(len(EbNodB_range))
BER=np.zeros(len(EbNodB_range))
MAX_ERR=30



for i,EbNodB in enumerate(EbNodB_range):

  count_noterr=0
  count_all=0
  count_err=0
  count_ball=0
  count_berr=0
  
  while count_err<MAX_ERR:

    sourcecode,codeword_1D=generate_codeword(N,Gres) #1D-array codeword
    codeword=codeword_1D[...,np.newaxis] 
    codeword=np.transpose(codeword) #2D-array codeword

    RX_BPSK=AWGN_channel(EbNodB,codeword)

    #受信信号の1列ごとの受け渡し(数値をx軸のみに変更)
    y=RX_BPSK[0,:].real #1D-array codeword
    #LLR
    EbNo = 10 ** (EbNodB / 10)
    No=1/EbNo
    Lc=4*y/No
    t=0
    EST_codeword=np.full(N,-1)
    SC_decoding(Lc)
    #print(codeword,EST_codeword)
    #復号をしない場合
    #EST_codeword=np.sign(y)
    #EST_codeword[EST_codeword==1]=0
    #EST_codeword[EST_codeword==-1]=1
    #EST_codeword=(EST_codeword@Gres)%2

    #calculate block error rate
    #print(sourcecode,EST_codeword)
    if np.any(sourcecode!=EST_codeword):#BLOCK error check
      count_err+=1
    
    count_all+=1

    #calculate bit error rate 
    count_berr+=np.sum(sourcecode!=EST_codeword)
    count_ball+=K

    print("\r","count_all=",count_all,",count_err=",count_err,"count_ball="\
          ,count_ball,"count_berr=",count_berr,end="")

  print("\n",EbNodB,"BLER=",count_err/count_all,"BER=",count_berr/count_ball)

  BLER[i]=count_err/count_all
  BER[i]=count_berr/count_ball

  if count_err/count_all<10**-5:
    print("finish")
    break

#output "BLER"

filename="polarLLR_{}_{},SN_des={}".format(N,K,design_SNR)

with open(filename,'w') as f:

    print("#N="+str(N),file=f)
    print("#K="+str(K),file=f)
    print("#EsNodB,BLER,BER",file=f)      #この説明はプログラムによって変えましょう！！！！！！！
    for i in range(len(EbNodB_range)):
        print(str(EbNodB_range[i]),str(BLER[i]),str(BER[i]),file=f)
'''

