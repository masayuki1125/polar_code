# -*- coding: utf-8 -*-
"""Polar code

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17nSDygyf1ulqlEL8ysCDYFsfF0eO_oNE
"""

#必要なライブラリ、定数
import numpy as np
import random
import time
import math
import copy
from decimal import *
random.seed(time.time())
#below is the info of basemat
N=1024 #log_2(N)==int <-- requirement
K=512
design_SNR=1

itr_num=np.log2(N)
itr_num=itr_num.astype(int)

G2=np.array([[1,0],[1,1]],dtype=np.int)

def tensordot(A):
  tmp0=np.zeros((A.shape[0],A.shape[1]),dtype=np.int)
  tmp1=np.append(A,tmp0,axis=1)
  #print(tmp1)
  tmp2=np.append(A,A,axis=1)
  #print(tmp2)
  tmp3=np.append(tmp1,tmp2,axis=0)
  #print(tmp3)
  return tmp3

Gres=G2
for i in range(itr_num-1):
  #print(i)
  Gres=tensordot(Gres)
print(Gres)
print(Gres.shape)

#frozen_bitの選択
'''
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
  res=indices_of_elements(z,N-K)

  return res

def indices_of_elements(v,l):
  tmp=np.argsort(v)[::-1]
  print(tmp)
  res=tmp[0:l]
  return res
'''

def generate_codeword(N,Gres):
  S=np.zeros(N)
  for i in range(N):
    S[i]=random.randrange(0,2)
    np.put(S,frozen_bit,0)
  U=(S@Gres)%2
  #print(U.shape)
  return S,U

#input "codeword"(2D(M,:)-array) 

def AWGN_channel(EbNodB,codeword):
  #modulation-channel
  # 送信側アンテナ数
  M =1
  # 受信側アンテナ数
  N = 1
  # 送信ビット列
  TX_bit =copy.deepcopy(codeword)
  # 送信ビット数
  bit_num =codeword.shape[1]

  # Additive Gaussian White Noiseの生成する際のパラメータ設定
  EbNo = 10 ** (EbNodB / 10)
  No=1/EbNo #Eb=1(fixed)

  # 0 -> 1, 1 -> -1としてBPSK変調
  TX_BPSK = TX_bit
  TX_BPSK[TX_bit==1]=-1
  TX_BPSK[TX_bit==0]=1

  # AWGN雑音の生成
  noise = np.random.normal(0, np.sqrt(No / 2), (M, bit_num)) \
          #+ 1j * np.random.normal(0, np.sqrt(No / 2), (M, bit_num))

  # AWGN通信路 = 送信シンボル間干渉が生じないような通信路で送信
  RX_BPSK = TX_BPSK + noise

  # 以下のprint関数の出力を表示すると、Noとほぼ一致するはず
  #print(np.dot(noise[0, :], np.conj(noise[0, :]))/bit_num)
  
  return RX_BPSK

#output "RX_BPSK"(2D(M,len(codeword)))-array

#SC復号

#node operation

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


def SC_decoding(a):
  global t
  #interior node operation
  if a.shape[0]==1:
    #frozen_bit or not
    if np.any(frozen_bit==t):
      tmp0=np.zeros(1)
    elif a>=0:
      tmp0=np.zeros(1)
    elif a<0:
      tmp0=np.ones(1)
    else:
      print("err!")
      exit()
    EST_codeword[t]=tmp0
    #print(t)
    t+=1
    #if t>=N:
      #exit()
    return tmp0

  #step1 left input a output u1_hat
  tmp1=np.split(a,2)
  f_half_a=chk(tmp1[0],tmp1[1])
  u1=SC_decoding(f_half_a)

  #step2 right input a,u1_hat output u2_hat 
  tmp2=np.split(a,2)
  g_half_a=tmp2[1]+(1-2*u1)*tmp2[0] 
  u2=SC_decoding(g_half_a)
  
  #step3 up input u1,u2 output a_hat
  res=np.concatenate([(u1+u2)%2,u2])
  return res

#frozen bit を定める

#frozen_bit=Bhattacharyya_bounds(N,K,design_SNR)
#np.savetxt("frozen_bit.txt",frozen_bit)
#sourcecode,codeword_1D=generate_codeword(N,Gres) #1D-array codeword
#codeword=codeword_1D[...,np.newaxis] 
#codeword=np.transpose(codeword) #2D-array codeword

#RX_BPSK=AWGN_channel(1,codeword)

#受信信号の1列ごとの受け渡し(数値をx軸のみに変更)
#y=RX_BPSK[0,:].real #1D-array codeword

#t=0
#EST_codeword=np.zeros(N)
#SC_decoding(y)

#EST_codeword=np.sign(y)
#EST_codeword[EST_codeword==1]=0
#EST_codeword[EST_codeword==-1]=1
#EST_codeword=(EST_codeword@Gres)%2
#print(sourcecode)
#print(EST_codeword)
#print(np.sum(sourcecode!=EST_codeword))
#print(sourcecode)
#print(EST_codeword)
#print(np.any(frozen_bit==500))

#input "PC_MATRIX"(2D(M,:)-array)


EbNodB_range=np.arange(-100,3.5,0.5)
BLER=np.zeros(len(EbNodB_range))
BER=np.zeros(len(EbNodB_range))
MAX_ERR=5

#frozen bit を定める
frozen_bit=np.loadtxt('frozen_bits512_1dB.txt',dtype=int)

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
    print(sourcecode,EST_codeword)
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
