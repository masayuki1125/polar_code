//
//  main.c
//  Bhattacharyya-bounds
//
//  Created by 池谷恒亮 on 2020/10/31.
//
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define N 1024
#define K 512

void make_bitrev(int n,int *bitrev);
void indices_of_greatest_elements(int num,int *Bb,double *z,int *bitrev);

int main(void) {
    
    //変数のメモリ割り当て
    double *z;
    int *bitrev,*Bb;
    double snr_db,S,u;
    int n,J,o;
    z = (double *)calloc(N,sizeof(double));
    bitrev = (int *)calloc(N,sizeof(int));
    Bb = (int *)calloc(N,sizeof(int));


    FILE*fp1;
    fp1 = fopen("frozen_bits_5dB.txt","w");



    make_bitrev(N,bitrev);

    n = log2(N);
    snr_db = 5;//defined as Es/N0
    S = pow(10,snr_db/10);


    z[0] = exp(-1*S);for(int i=1;i<n+1;i++){
        J = pow(2,(double)i);
        o = J/2;
        for(int j=0;j<o;j++){
            u = z[j];
            z[j] = (2*u) - (u*u) ;
            z[j+o] = u*u;
        }
    }

    for(int i=0;i<N;i++){
        printf("z[]=%f\n",z[i]);
    }

    indices_of_greatest_elements(N-K,Bb,z,bitrev);
    for(int i=0;i<N;i++){
        printf("Bb[]=%d\n",Bb[i]);
        fprintf(fp1,"%d\n",Bb[i]);
    }
    fclose(fp1);
    //    fclose(fp2);
    //printf("%lf\n",phi(0));
    printf("Hello, World!\n");
    return 0;
}
void make_bitrev(int n,int *bitrev){
    int i, j, k, n2;
    n2 = n / 2;  i = j = 0;
    for ( ; ; ) {
        bitrev[i] = j;
        if (++i >= n) break;
        k = n2;
        while (k <= j) {  j -= k;  k /= 2;  }
        j += k;
    }
}
void indices_of_greatest_elements(int num,int *Bb,double *z,int *bitrev){
//    FILE*fp2;
//    fp2 = fopen("polarization_Bb_3.txt","w");
    double temp0,temp1;
    double sort[N];
   for(int i=0;i<N;i++){
        sort[i] = z[i];
        Bb[i] = i;
    }
    
    for(int i=0;i<N;i++){
        for(int j=i+1;j<N;j++){
            if(sort[i]<sort[j]){
                temp0 = sort[i];
                sort[i] = sort[j];
                sort[j] = temp0;
                temp1 = Bb[i];
                Bb[i] = Bb[j];
                Bb[j] = temp1;
            }
        }
    }
//    for(int i=0;i<N;i++){
//        printf("z[%d]=%e\n",i,sort[i]);
//        fprintf(fp2,"%d,%lf\n",i,sort[i]);
//    }
//    for(int i=0;i<N;i++){
//        Bb[i] = bitrev[Bb[i]];
//        }
//    fclose(fp2);   
}