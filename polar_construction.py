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


# In[4]:


class Improved_GA(Improved_GA):
  def reverse(self,index,n):
    '''
    make n into bit reversal order
    '''
    tmp=format (index,'b')
    tmp=tmp.zfill(n+1)[:0:-1]
    res=int(tmp,2) 
    return res


# In[5]:


class Improved_GA(Improved_GA):
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
      print(gamma)
      print(zeta)
      print("gamma is - err")

    return gamma


# In[6]:


class Improved_GA(Improved_GA):
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
      print(a)
      print(b)
      print(gamma)
      print("gamma is - err")    
      
    if gamma==0.0:
      print("gamma is underflow")
      print(gamma)
      print(zeta) 

    return gamma


# In[7]:


'''
class Improved_GA(Improved_GA):
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
'''


# In[8]:


class Improved_GA(Improved_GA):
  def maxstr(self,a,b):
    def f(c):
      return np.log(1+np.exp(-1*c))
    return max(a,b)+f(abs(a-b))


# In[9]:


class Improved_GA(Improved_GA):
  def left_operation(self,gamma1,gamma2):
    
    #calc zeta
    zeta1=self.xi(gamma1)
    zeta2=self.xi(gamma2)
    
    if gamma1<=self.G_0 and gamma2<=self.G_0:
        
      sq=1/2*gamma1*gamma2
      cu=-1/4*gamma1*(gamma2**2)-1/4*(gamma1**2)*gamma2
      fo=5/24*gamma1*(gamma2**3)+1/4*(gamma1**2)*(gamma2**2)+5/24*(gamma1**3)*gamma2
      
      gamma=sq+cu+fo
      
            
    else:
      
      gamma=self.xi_inv(zeta1+math.log(2-math.e**zeta1)) 
      #tmp=self.maxstr(zeta1,zeta2)
      #zeta=self.maxstr(tmp,zeta1-zeta2)

      #gamma=self.xi_inv(zeta)
      
      #gamma=inv_phi(1-(1-phi(gamma1))*(1-phi(gamma2)))
      #print("1")
    return gamma
          
  def right_operation(self,gamma1,gamma2):
    #print("0")
    return gamma1+gamma2
  
  
  def main_const(self,N,K,high_des,beta=1000,ind_high_des=False,ind_low_des=False):
    
    n=np.log2(N).astype(int)
    gamma=np.zeros((n+1,N)) #matrix
    
    #初期値の代入
    if beta==1000:
      gamma[0,:]=4*(10 ** (high_des / 10))
      
    else:
      gamma[0,ind_high_des]=4*(10 ** (high_des / 10))
      gamma[0,ind_low_des]=(beta**2)*4*(10 ** (high_des / 10))
    
    for i in range(1,gamma.shape[0]):
      for j in range(gamma.shape[1]):
        if (j//2**(n-i))%2==0:
          gamma[i,j]=self.left_operation(gamma[i-1,j],gamma[i-1,j+2**(n-i)])
        
        else :
          gamma[i,j]=self.right_operation(gamma[i-1,j],gamma[i-1,j-2**(n-i)])
    
    tmp=np.argsort(gamma[n,:])
    
    frozen_bits=np.sort(tmp[:N-K])
    info_bits=np.sort(tmp[N-K:])
    
    return frozen_bits, info_bits


# In[10]:


class GA():
    
  def main_const(self,N,K,high_des,beta=1000,ind_high_des=False,ind_low_des=False):
    
    
    
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
    
    if beta==1000:
      gamma[0,:]=4*(10 ** (high_des / 10))
      
    else:
      gamma[0,ind_high_des]=4*(10 ** (high_des / 10))
      gamma[0,ind_low_des]=(beta**2)*4*(10 ** (high_des / 10))
    
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


# In[11]:


class GA(GA):
  
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


# In[12]:


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
    


# In[13]:


class inv_GA(inv_GA):
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


# In[14]:


class inv_GA(inv_GA):
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


# In[15]:


class monte_carlo():
      
  def main_const(self,N,K_res,high_des,beta=1000,ind_high_des=False,ind_low_des=False):
    
    '''
    import ray
    ray.init()
    

    
    result_ids=[]
        
    for _ in range(10):
      #multiprocess    
      result_ids.append(output.remote(N,high_des,beta=1000,ind_high_des=False,ind_low_des=False))  # 並列演算
      #resultは長さ1のリストの中にBLER,BERの2つのarrayのtupleが入った配列
    
    result=ray.get(result_ids)
    
    c_all=np.zeros(N)
    for k in range(10):
      tmp1=result[k]
      c_all+=tmp1

    print(c_all)
    '''
    c=self.output(N,high_des,beta=1000,ind_high_des=False,ind_low_des=False)
    
    tmp=np.argsort(c)
    frozen_bits=np.sort(tmp[:N-K_res])
    info_bits=np.sort(tmp[N-K_res:])
    
    return frozen_bits,info_bits
    #return result


# In[16]:


class monte_carlo(monte_carlo):
#@ray.remote
  def output(self,N,high_des,beta=1000,ind_high_des=False,ind_low_des=False):
    
    #あるSNRで計算結果を出力する関数を作成
    #cd.main_func must input 'EbNodB' and output 1D 'codeword' and 'EST_codeword'

    from polar_code import polar_code

    self.pc=polar_code(N,0)
    #prepere constant
    c=np.zeros(N)
    M=5*10**4
    
    #seed値の設定
    #np.random.seed()

    for _ in range(M):
      _,codeword=self.pc.polar_encode()
      
      if beta==1000:
        Lc=-1*self.pc.ch.generate_LLR(codeword,high_des)#デコーダが＋、ー逆になってしまうので-１をかける
      else:#for 4PAM NOMA
        [cwd1,cwd2]=np.split(codeword,2)
        const1=self.pc.ch.generate_QAM(cwd1)
        const2=self.pc.ch.generate_QAM(cwd2)
        res_const=beta*const1*(-1*const2)+const2
        
        high_des = 10 ** (high_des / 10)
        No=1/high_des
        RX_const=self.pc.ch.add_AWGN(res_const,No)
        RX_const=RX_const.real #In_phaseのみ取り出す
        tmp_Lc=-1*self.calc_LLR(RX_const,No,beta)
        
        Lc=np.zeros(len(tmp_Lc))
        Lc[ind_low_des]=tmp_Lc[0:len(tmp_Lc)//2]
        Lc[ind_high_des]=tmp_Lc[len(tmp_Lc)//2:len(tmp_Lc)]
        
      llr=self.pc.polar_decode(Lc) 
        
      d=np.zeros(len(llr))
        #print(llr)
        #from IPython.core.debugger import Pdb; Pdb().set_trace()
      d[llr<1]=0
      d[llr>=1]=1
      c=c+d

    return c


# In[17]:


class monte_carlo(monte_carlo):
  @staticmethod
  def calc_exp(x,A,No):
    #解が0にならないように計算する
    res=np.zeros(len(x))
    for i in range(len(x)):
      if (x[i]-A)**2/No<30:
        res[i]=np.exp(-1*(x[i]-A)**2/No)
      else:
        res[i]=10**(-15)
    return res
  
  def calc_LLR(self,x,No,beta):
    A1=self.calc_exp(x,-1-beta,No)
    A2=self.calc_exp(x,-1+beta,No)
    A3=self.calc_exp(x,1-beta,No)
    A4=self.calc_exp(x,1+beta,No)
    
    y2=np.log((A3+A4)/(A1+A2))
    y1=np.log((A2+A3)/(A1+A4))
    #print(y2)
    #print(y1)
    
    return np.concatenate([y1,y2])


# In[18]:



if __name__=="__main__":
  N=512
  K=256
  design_SNR=0
  beta=0.1
  const=monte_carlo()
  f,i=const.main_const(N,K,design_SNR,beta,np.arange(N//2,N),np.arange(0,N//2))
  np.savetxt("i",i,fmt='%i')
  


# In[90]:


#check
if __name__=="__main__":
  N=512
  const1=Improved_GA()
  frozen1,info1=const1.main_const(N,N//2,0)
  #print(np.sum(frozen1!=a))
  print(frozen1)
  
  const2=GA()
  frozen2,info2=const2.main_const(N,N//2,0,0.5,np.arange(N//2,N),np.arange(0,N//2))
  
  print(np.any(frozen1!=f))
  
  #const3=inv_GA()
  #low,high=const3.main_const(N,frozen2,info2,np.arange(N//2,N),np.arange(0,N//2))
  
  a=np.any(frozen1!=frozen2)
  #print(a)


# In[ ]:


if __name__=="__main__":
  N=1024
  K=N//2
  const1=Improved_GA()
  f,i=const1.main_const(N,K,0,-10,np.arange(0,K),np.arange(K,N))

  const2=GA()
  f2,i2=const2.main_const(N,K,0,-10,np.arange(0,K),np.arange(K,N))
  


# In[ ]:


if __name__=="__main__":
  N=1024
  K=N//2
  const2=Improved_GA()
  const2.main_const_for_different_SNR(N,K,-10,0)
  #frozen2,info2=const2.main_const(N,N//2,0,)
  
  #info2=np.arange(0,N//2)
  #frozen2=np.arange(N//2,N)
  #print(len(frozen2))
  #print(len(info2))
  
  #const3=inv_GA()
  #low,high=const3.main_const(N,frozen2,info2)
  
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

