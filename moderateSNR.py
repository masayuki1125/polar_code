#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np
import cupy as cp
pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
cp.cuda.set_allocator(pool.malloc)
from polar_code import polar_code 
from AWGN import _AWGN
from scipy.stats import norm
import pickle
import ray 
import multiprocessing
import calc_capacity as cc


# In[2]:


ray.init()


# In[3]:


@ray.remote
def output(dumped,EbNodB,target_BLER,MAX_parallel):
  '''
  #あるSNRで計算結果を出力する関数を作成
  target_BLERの100倍の反復を行う
  #cd.main_func must input 'EbNodB' and output 1D 'codeword' and 'EST_codeword'
  '''

  #de-seriallize file
  cd=pickle.loads(dumped)
  #seed値の設定
  np.random.seed()

  #prepare some constants
  MAX_ALL=int(100/(target_BLER*MAX_parallel))
  #MAX_BITERR=10**1
  count_bitall=0
  count_biterr=0
  count_all=0
  count_err=0
  

  while count_all<MAX_ALL:# and count_err<MAX_BITERR:
    #print("\r"+str(count_err),end="")
    information,EST_information=cd.main_func(EbNodB)
    
    #calculate block error rate
    if np.any(information!=EST_information):
        count_err+=1
    count_all+=1

    #calculate bit error rate 
    count_biterr+=np.sum(information!=EST_information)
    count_bitall+=len(information)

  return count_err,count_all,count_biterr,count_bitall


# In[ ]:





# In[4]:


class MC():
  def __init__(self):
    self.MAX_parallel=multiprocessing.cpu_count()

  #特定のNに関する出力
  def monte_carlo_get_ids(self,dumped,EsNodB,target_BLER):
    '''
    input:main_func
    -----------
    dumped:seriallized file 
    main_func: must input 'EbNodB' and output 1D 'codeword' and 'EST_codeword'
    -----------
    output:result_ids(1Darray x:MAX_parallel)

    '''
    result_ids=[]

    for _ in range(self.MAX_parallel):
      #multiprocess    
      result_ids.append(output.remote(dumped,EsNodB,target_BLER,self.MAX_parallel))  # 並列演算
      #resultは長さ1のリストの中にBLER,BERの2つのarrayのtupleが入った配列

    return result_ids
  
  def monte_carlo_calc(self,result_ids):

    #prepare constant

    result=ray.get(result_ids)
    #resultには同じSNRのリストが入る
    count_err=0
    count_all=0
    count_biterr=0
    count_bitall=0
    
    for i in range(self.MAX_parallel):
      tmp1,tmp2,tmp3,tmp4=result[i]
      count_err+=tmp1
      count_all+=tmp2
      count_biterr+=tmp3
      count_bitall+=tmp4

    BLER=count_err/count_all
    BER=count_biterr/count_bitall
    
    return BLER,BER



# In[ ]:


def liner_approx(K_list,BLER_list,target_BLER):
  log_target_BLER=np.log10(target_BLER)
  log_BLER=np.log10(BLER_list)
  print(K_list,log_BLER)
  #回帰直線
  linear=np.polyfit(K_list,log_BLER,1)
  print(linear)
  target_K=(log_target_BLER-linear[1])/linear[0]
  
  if target_K<0:
    target_K=0

  return target_K


# In[11]:
def est_InfoRate(N,EsNodB,beta,Rayleigh):
  print("beta=",beta)
  #set constant
  EsNo=10 ** (EsNodB / 10)
  target_BLER=10**-3

  C_infinite=cc.mutual_info(EsNodB,beta,Rayleigh)
  print("C_inf=",C_infinite)
  C=max(0.01,cc.finite_bound(N,target_BLER,EsNo,C_infinite))
  print("C=",C)

  #decide K
  K=int(N*C)
  
  K=est_InfoLength(N,K,EsNodB,beta,target_BLER,Rayleigh)
  print(K)
  
  return K/N

def est_BLER(EsNodB,dumped,target_BLER):
  '''
  特定のN,K,EsNodB,betaに対して、推定BLERを算出する関数
  '''
  mc=MC()
  
  result_ids=mc.monte_carlo_get_ids(dumped,EsNodB,target_BLER)

  BLER,BER=mc.monte_carlo_calc(result_ids)
  
  return BLER,BER

def est_InfoLength(N,K_init,EsNodB,beta,target_BLER,Rayleigh):
  K_res=K_init
  itr_num=0
  MAX_ITR=5
  K_list=np.zeros(MAX_ITR)
  BLER_list=np.ones(MAX_ITR)
  while itr_num<MAX_ITR:
    #initialize channel coding 
    print(N,K_res,beta,Rayleigh)
    cd=polar_code(N,K_res,beta,Rayleigh)
    dumped=pickle.dumps(cd)
    
    #calcurate BLER
    BLER,_=est_BLER(EsNodB,dumped,target_BLER)
    print((N,K_res),BLER)  
    if K_res==1 and BLER>target_BLER*10:
      K_res=0
      return K_res  
    
    #測定範囲はtarget_BLER*100>x>target_BLER/100
    if BLER==0.0 or BLER>target_BLER*100:
      #測定外のときは1割変更してもう一度測定
      threshold=int(K_res*0.1)
      if threshold==0:
        threshold=5
      itr_num-=1
    else:
      #最大6％情報長が変更される
      threshold=int(K_res*0.05*abs(math.log10(BLER/target_BLER)))
      if threshold==0:
        threshold=int(K_res*0.3*abs(math.log10(BLER/target_BLER))+1)
      #get BLER and K_res 
      K_list[itr_num]=K_res
      BLER_list[itr_num]=BLER

    print(threshold)
    change=min(100,threshold)
    
    #change K_res
    if BLER>target_BLER:
      if K_res-change<=0:
        K_res=1
        #処理がスタックしないようにする
        itr_num+=1
        
      else:
        K_res-=change
    else:
      if K_res+change>=N:
        K_res=N-1
      else:
        K_res+=change
    
    print("K=",K_res)
        
    itr_num+=1
    print(itr_num)
  
  print(K_list,BLER_list)
  K_res=liner_approx(K_list,BLER_list,target_BLER)
      
  return K_res

# In[6]:
EsNodB2=0
data_num=10
for n in [256,512]:
  for EsNodB1 in [3,5,10]:
    R1=np.zeros(data_num)
    R2=np.zeros(data_num)
    
    for i in range(1,data_num-1):  
      print(EsNodB2)
      R1[i]=est_InfoRate(n,EsNodB1,i/data_num,False)
      R2[i]=est_InfoRate(n,EsNodB2,i/data_num,True)
      print("R1,R2=",R1[i],R2[i])
    
    filename="Rate_polar_{}_{}_{}".format(EsNodB1,EsNodB2,n)
    with open(filename,'w') as f:

      print("#R1,R2",file=f)  
      for i in range(data_num):
        print(str(R1[i]),str(R2[i]),file=f)