#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import math
from decimal import *
import sympy as sp


# In[3]:


class Improved_GA():
  
  def __init__(self):
    #for construction(GA)
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

  def reverse(self,index,n):
    '''
    make n into bit reversal order
    '''
    tmp=format (index,'b')
    tmp=tmp.zfill(n+1)[:0:-1]
    res=int(tmp,2) 
    return res

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

  def bisection_method(self,zeta):
  
    #set constant
    min_num=self.G_2
    max_num=-4*(zeta-1/2*math.log(math.pi))
    error_accept=1/100

    def f(x):
      zeta=-1*x/4+math.log(math.pi)/2-math.log(x)/2+math.log(1-(math.pi**2)/(4*x)+self.K_0/(x**2))
      return zeta

    #initial value
    a=min_num
    b=max_num
    error=b-a

    #very small zeta situation
    if f(max_num)>zeta:
      print("error")
      #gamma=max_num

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
      
    if gamma==0.0:
      print("gamma is underflow")
      print(gamma)
      print(zeta) 

    return gamma

  def main_const(self,N,K,design_SNR,bit_reverse=True):
    #make n where 2**n=N
    n=np.log2(N).astype(int)
    
    gamma=np.zeros(N)
     
    gamma[0]=4*(10 ** (design_SNR / 10)) #mean of LLR when transmit all 0
    for i in range(1,n+1):
      J=2**(i-1)
      for j in range(0,J):
        u=gamma[j]
        if u<=self.G_0:
          gamma[j]=(u**2)/2-(u**3)/2+2*(u**4)/3
        else:
          z=self.xi(u)
          gamma[j]=self.xi_inv(z+math.log(2-math.e**z))
        
        gamma[j+J]=2*u

    tmp=self.indices_of_elements(gamma,N)
    frozen_bits=np.sort(tmp[:N-K])
    info_bits=np.sort(tmp[N-K:])

    if bit_reverse==True:
      for i in range(len(frozen_bits)):
        frozen_bits[i]=self.reverse(frozen_bits[i],n)
      frozen_bits=np.sort(frozen_bits)

      for i in range(len(info_bits)):
        info_bits[i]=self.reverse(info_bits[i],n)
      info_bits=np.sort(info_bits)

    return frozen_bits,info_bits

  @staticmethod
  def indices_of_elements(v,l):
    tmp=np.argsort(v)
    res=tmp[0:l]
    return res


# In[8]:


class GA():
    
  def main_const(self,N,K,low_des,high_des=1000,ind_low_des=False,ind_high_des=False,bit_reverse=False):
    
    
    
    #make n where 2**n=N
    n=np.log2(N).astype(int)
    
    #O(N**2) complexity
    #constant for GA operation
    a=-0.4527
    b=0.0218
    c=0.86
    G=10
    
    def phi(gamma):   
      if gamma<=G:
        zeta=math.exp(a*gamma**c+b)
        
      else:
        zeta=(math.pi/gamma)**(1/2)*math.exp(-gamma/4)*(1-10/(7*gamma))
      
      return zeta
    
    Z=phi(G)
    
    def inv_phi(zeta):
      if zeta>=Z:
        gamma=((math.log(zeta)-b)/a)**(1/c)
      else:
        gamma=self.bisection_method(zeta)
    
      return gamma
    
    
    def left_operation(gamma1,gamma2):
      
      #calc zeta
      zeta1=phi(gamma1)
      zeta2=phi(gamma2)
           
      d1=Decimal("1")
      d2=Decimal(zeta1)
      d3=Decimal(zeta2)
      
      zeta=d1-(d1-d2)*(d1-d3)
      #print(zeta)
      
      #for underflow
      if zeta==0:
        zeta=10**(-50)

      gamma=inv_phi(zeta)
      
      #gamma=inv_phi(1-(1-phi(gamma1))*(1-phi(gamma2)))
      #print("1")
      return gamma
            
    def right_operation(gamma1,gamma2):
      #print("0")
      return gamma1+gamma2
    
    #main operation
    
    gamma=np.zeros((n+1,N)) #matrix
    
    if high_des==1000:
      print("itiyoubunnpu")
      gamma[0,:]=4*(10 ** (low_des / 10))
      
    else:
      gamma[0,ind_low_des]=4*(10 ** (low_des / 10))
      gamma[0,ind_high_des]=4*(10 ** (high_des / 10))
    
    for i in range(1,gamma.shape[0]):
      for j in range(gamma.shape[1]):
        if (j//2**(n-i))%2==0:
          gamma[i,j]=left_operation(gamma[i-1,j],gamma[i-1,j+2**(n-i)])
        
        else :
          gamma[i,j]=right_operation(gamma[i-1,j],gamma[i-1,j-2**(n-i)])
    
    tmp=np.argsort(gamma[n,:])
    
    frozen_bits=np.sort(tmp[:N-K])
    info_bits=np.sort(tmp[N-K:])
    
    '''
    削除予定
    if bit_reverse==True:
      for i in range(len(frozen_bits)):
        frozen_bits[i]=self.reverse(frozen_bits[i])
      frozen_bits=np.sort(frozen_bits)

      for i in range(len(info_bits)):
        info_bits[i]=self.reverse(info_bits[i])
      info_bits=np.sort(info_bits)
    '''

    return frozen_bits,info_bits
  
  def bisection_method(self,zeta):
      
    #set constant
    
    min_num=10
    max_num=-4*math.log(zeta)
    error_accept=10**(-10)

    def f(x):
      zeta=(math.pi/x)**(1/2)*math.exp(-x/4)*(1-10/(7*x))
      return zeta

    #initial value
    a=min_num
    b=max_num
    error=b-a

    #very small zeta situation
    if f(max_num)>zeta:
      print("error")
      #gamma=max_num

    count=0
    while error>error_accept:
      count+=1
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
      
    if gamma==0.0:
      print("gamma is underflow")
      print(gamma)
      print(zeta) 

    return gamma


# In[10]:


class inv_GA():
  
  def __init__(self):
    self.a=-0.4527
    self.b=0.0218
    self.c=0.86
    self.G=10
    self.Z=self.phi(self.G)
  
  def phi(self,gamma):   
    if gamma<=self.G:
      zeta=math.exp(self.a*gamma**self.c+self.b)
    else:
      zeta=(math.pi/gamma)**(1/2)*math.exp(-gamma/4)*(1-10/(7*gamma))
    
    return zeta
  
  def inv_phi(self,zeta):
    if zeta>=self.Z:
      gamma=((math.log(zeta)-self.b)/self.a)**(1/self.c)
    else:
      gamma=self.bisection_method(zeta)
    return gamma 
  
  def main_const(self,N,frozen_bits,info_bits):
    
    #make n where 2**n=N
    n=np.log2(N).astype(int)
    
    zero=1
    inf=10000
    
    gamma=np.zeros((n+1,N)) #matrix
    
    #一番下の行に代入
    gamma[n,frozen_bits]=zero
    gamma[n,info_bits]=inf
    
    def left_operation(gamma1,gamma2):
      #C=phi(gamma1)
      #gamma2-x=x'(x+x'=gamma2)
      #x=gamma
      #res1=x
      #res2=A-x
      #A=gamma2
      
      #calc zeta
      #print(gamma1)
      
      zeta=self.phi(gamma1)
      #if gamma2<1: #しきい値1は適当に設定した。
        #逆関数から計算する
        #if f(gamma2/2,gamma2)<zeta:
          #取りうる値のペアではなかったとき、取りうる値の中で最小の値を出力
          #res1=gamma2/2
        
        #else:
          #res1=self.res.subs([(self.x, gamma1),(self.A, gamma2)])
      
      #else:
      res1=self.bisection_method_for_inv_GA(zeta,gamma2)
      
      if np.random.randint(2,size=1)==0:
        res2=gamma2-res1
       
      res2=gamma2-res1
      #res1<res2 と仮定した
      
      if res1<0 or res2<0:
        print("res minus error")
      
      return res1,res2
    
    '''削除予定        
    def right_operation(gamma1,gamma2):
      
      res=gamma1-gamma2
      
      if res<0:
        print("right_operation error")
    
      #print("0")
      return res
    '''
    
    #inv_GA process
    for i in reversed(range(0,gamma.shape[0]-1)):
      for j in range(gamma.shape[1]):
        if (j//2**(n-i-1))%2==0:
          gamma[i,j],gamma[i,j+2**(n-i-1)]=left_operation(gamma[i+1,j],gamma[i+1,j+2**(n-i-1)])
      
        #print(i,j) 
        #print(gamma[i,j])
    
    print(gamma[0,:])
    #gamma[0,:]が大きいほど、信号点配置が大きくなければならない
    tmp=np.argsort(gamma[0,:])
    low_power_bits=np.sort(tmp[:N//2])
    high_power_bits=np.sort(tmp[N//2:])
    
    return low_power_bits,high_power_bits
    
  def bisection_method(self,zeta):
      
    #set constant
    min_num=10
    max_num=-4*math.log(zeta)
    error_accept=10**(-10)

    def f(x):
      zeta=(math.pi/x)**(1/2)*math.exp(-x/4)*(1-10/(7*x))
      return zeta

    #initial value
    a=min_num
    b=max_num
    error=b-a

    #very small zeta situation
    if f(max_num)>zeta:
      print("error1")
      #gamma=max_num

    count=0
    while error>error_accept:
      count+=1
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
      
    if gamma==0.0:
      print("gamma is underflow")
      print(gamma)
      print(zeta) 

    return gamma
  
  def bisection_method_for_inv_GA(self,zeta,A):
    #増大関数について考える
      
    #set constant
    min_num=0
    max_num=A/2
    error_accept=10**(-10)

    def f(x):
      zeta=self.phi(x)+self.phi(A-x)-self.phi(x)*self.phi(A-x)
      #if zeta>1:
        #print("zeta error")
        #print(zeta)
        
      return zeta

    #initial value
    a=min_num
    b=max_num
    error=b-a

    #very small zeta situation
    if f(max_num)>zeta:
      #取りうる値のペアではなかったとき、取りうる値の中で最小の値を出力
      return A/2
      #print("error2")
      #print(zeta)
      #print(f(max_num))
      #gamma=max_num

    count=0
    while error>error_accept:
      count+=1
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
      
    if gamma==0.0:
      print("gamma is underflow")
      print(gamma)
      print(zeta) 

    return gamma


# In[14]:


#check
if __name__=="__main__":
  N=2048
  const1=Improved_GA()
  frozen1,info1=const1.main_const(N,N//2,0)
  
  const2=GA()
  frozen2,info2=const2.main_const(N,N//2,0)
  
  print(np.any(frozen1!=frozen2))
  
  const3=inv_GA()
  low,high=const3.main_const(N,frozen2,info2)
  
  #a=np.any(frozen1!=frozen2)
  #print(a)


# In[ ]:


if __name__=="__main__":
  
  const2=GA()
  frozen2,info2=const2.main_const(N,N//2,0,)
  
  info2=np.arange(0,N//2)
  frozen2=np.arange(N//2,N)
  print(len(frozen2))
  print(len(info2))
  
  const3=inv_GA()
  low,high=const3.main_const(N,frozen2,info2)
  
  count=0
  
  while True:
    tmp=low
    
    frozen2,info2=const2.main_const(N,N//2,-10,0,low,high)
    low,high=const3.main_const(N,info2,frozen2)
    
    if np.all(tmp==low):
      break
    
    count+=1
    print(count)
    
  print(frozen2)

