#!/usr/bin/env python
# coding: utf-8

# In[56]:


import numpy as np
import math
from decimal import *
from AWGN import _AWGN

ch=_AWGN()


# In[57]:


class coding():
  def __init__(self,N):
    super().__init__()

    self.N=N
    self.R=0.5
    self.K=math.floor(self.R*self.N)
    self.design_SNR=1

    #for construction
    self.G_0=0.2
    self.G_1=0.7
    self.G_2=10
    self.a0=-0.002706
    self.a1=-0.476711
    self.a2=0.0512
    self.a=-0.4527
    self.b=0.0218
    self.c=0.86
    self.K_0=8.554

    self.Z_0=self.xi(self.G_0)
    self.Z_1=self.xi(self.G_1)
    self.Z_2=self.xi(self.G_2)


    #for decoder
    self.n=0#calcurate codeword number
    self.k=0#calcurate information number
    self.EST_information=np.zeros(self.K)

    #prepere constants
    tmp2=np.log2(self.N)
    self.itr_num=tmp2.astype(int)
    self.bit_reversal_sequence=self.reverse_bits()
    
    #choose construnction
    #self.frozen_bits,self.info_bits=self.Bhattacharyya_bounds()
    self.frozen_bits,self.info_bits=self.Improved_GA()

    #self.Gres=self.make_H() #no need

    self.filename="polar_code_{}_{}".format(self.N,self.K)


# In[58]:


class coding(coding):
  def reverse_bits(self):
    res=np.zeros(self.N,dtype=int)

    for i in range(self.N):
      tmp=format (i,'b')
      tmp=tmp.zfill(self.itr_num+1)[:0:-1]
      #print(tmp) 
      res[i]=self.reverse(i)
    return res

  def reverse(self,n):
    tmp=format (n,'b')
    tmp=tmp.zfill(self.itr_num+1)[:0:-1]
    res=int(tmp,2) 
    return res


# In[59]:


class coding(coding):
   
  def xi(self,gamma):

    if gamma<=self.G_0:
      zeta=-1*gamma/2+(gamma**2)/8-(gamma**3)/8
    
    elif self.G_0<gamma<=self.G_1:
      zeta=self.a0+self.a1*gamma+self.a2*(gamma**2)

    elif self.G_1<gamma<self.G_2:
      zeta=self.a*(gamma**self.c)+self.b

    elif self.G_2<=gamma:
      zeta=-1*gamma/4+math.log(math.pi)/2-math.log(gamma)/2+math.log(1-(math.pi**2)/(4*gamma)+self.K_0/(gamma**2))
    
    if zeta>0:
      print("zeta is + err")

    return zeta

  def xi_inv(self,zeta):

    if self.Z_0<=zeta:
      gamma=-2*zeta+zeta**2+zeta**3

    elif self.Z_1<=zeta<self.Z_0:
      gamma=(-1*self.a1-(self.a1**2-4*self.a2*(self.a0-zeta))**(1/2))/(2*self.a2)

    elif self.Z_2<zeta<self.Z_1:
      gamma=((zeta-self.b)/self.a)**(1/self.c)

    elif zeta<=self.Z_2:
      gamma=self.bisection_method(zeta)
      #gamma=-4*zeta

    if gamma<0:
      print("gamma is - err")

    return gamma


# In[60]:


class coding(coding):
  def bisection_method(self,zeta):

    #set constant
    min_num=self.G_2
    max_num=10**7
    error_accept=1

    def f(x):
      zeta=-1*x/4+math.log(math.pi)/2-math.log(x)/2+math.log(1-(math.pi**2)/(4*x)+self.K_0/(x**2))
      return zeta

    #initial value
    a=min_num
    b=max_num
    error=b-a

    #very small zeta situation
    if f(max_num)>zeta:
      print(f(max_num))
      print(zeta)
      gamma=max_num

    else:

      while error>error_accept:
        c=(b+a)/2 #center value

        if f(c)>=zeta:
          a=c
          error=b-a
        
        elif f(c)<zeta:
          b=c
          error=b-a
        
        if error<0:
          print("something is wrong")
        #print("\r",error,end="")
      
      gamma=(b+a)/2

      if gamma<0:
        print("gamma is - err")

    return gamma


# In[61]:


class coding(coding):
    
  def Improved_GA(self,bit_reverse=True):
    gamma=np.zeros(self.N)
    
    gamma[0]=4*self.design_SNR
    for i in range(1,self.itr_num+1):
      J=2**(i-1)
      for j in range(0,J):
        u=gamma[j]
        if u<=self.G_0:
          gamma[j]=(u**2)/2-(u**3)/2+2*(u**4)/3
        else:
          z=self.xi(u)
          gamma[j]=self.xi_inv(z+math.log(2-math.e**z))
        
        gamma[j+J]=2*u
  
    tmp=self.indices_of_elements(gamma,self.N)
    frozen_bits=np.sort(tmp[:self.N-self.K])
    info_bits=np.sort(tmp[self.N-self.K:])

    if bit_reverse==True:
      for i in range(len(frozen_bits)):
        frozen_bits[i]=self.reverse(frozen_bits[i])
      frozen_bits=np.sort(frozen_bits)

      for i in range(len(info_bits)):
        info_bits[i]=self.reverse(info_bits[i])
      info_bits=np.sort(info_bits)

    return frozen_bits,info_bits

  @staticmethod
  def indices_of_elements(v,l):
    tmp=np.argsort(v)
    res=tmp[0:l]
    return res


# In[62]:


'''
class coding(coding):
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
'''


# In[63]:


'''
class coding(coding):
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
'''


# In[64]:


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
    codeword=self.encode(u_message[self.bit_reversal_sequence])
    #codeword=u_message@self.Gres%2
    return information,codeword


# In[65]:


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
        tmp0=np.zeros(1)
      
      else :
        self.EST_information[self.k]=a
        self.k+=1

        if a>=0:
          tmp0=np.zeros(1)
        elif a<0:
          tmp0=np.ones(1)
      
      self.n+=1

      return tmp0

    
    #tmp=np.reshape(a,[2,len(a)//2],order="F")
    tmp=np.split(a,2,axis=0)

    #step1 left input a output u1_hat
    f_half_a=self.chk(tmp[0],tmp[1])
    #f_half_a=tmp1[0]+tmp1[1]
    u1=self.SC_decoding(f_half_a)

    #step2 right input a,u1_hat output u2_hat 
    g_half_a=tmp[1]+(1-2*u1)*tmp[0] #
    u2=self.SC_decoding(g_half_a)
  
    #step3 up input u1,u2 output a_hat
    res=np.concatenate([(u1+u2)%2,u2])
    return res


# In[66]:


class decoding(decoding):
  def polar_decode(self,Lc):
    #initialize 
    self.n=0
    self.k=0

    self.SC_decoding(Lc)
    res=self.EST_information
    #err chenck
    if len(res)!=self.K:
      print("information length error")
      exit()
      
    res=-1*np.sign(res)
    EST_information=(res+1)/2

    return EST_information


# In[67]:


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


# In[69]:


if __name__=="__main__":

    N=2048
    pc=polar_code(N)
    #a,b=pc.main_func(1)
    #print(len(a))
    #print(len(b))
    def output(EbNodB):
      count_err=0
      count_all=0
      count_berr=0
      count_ball=0
      MAX_ERR=8

      while count_err<MAX_ERR:
        
        pc=polar_code(N)
        information,EST_information=pc.main_func(EbNodB)
      
        if np.any(information!=EST_information):#BLOCK error check
          count_err+=1
        
        count_all+=1

        #calculate bit error rate 
        count_berr+=np.sum(information!=EST_information)
        count_ball+=N

        print("\r","count_all=",count_all,",count_err=",count_err,"count_ball="              ,count_ball,"count_berr=",count_berr,end="")

      print("BER=",count_berr/count_ball)
      return  count_err,count_all,count_berr,count_all
    
    output(3)
    

