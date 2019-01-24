/*
 *	日期：	2019-1-24　
 *	时间：	14:42
 *	姓名：	杨丰拓
 */
//******************************************************************************************************************//
//对大量数据进行归约操作（例如，最大，最小，求和等）可通过使用共享内存缩短归约操作的时间．
//本程序预设目标归约2^20~2^30个数据，核划分为一维网格一维线程块，通过三次归约操作求出最大值．
//实际可处理数据为0~2^26.
//本程序可以处理0~2^20个数据，但实际上相对与处理的数据来说代码略有多余．在处理0~2^10个数据时仅需调用一次核函数，
//在处理2^10~2^20个数据时仅需调用两次核函数．当需要处理的数据量达到2^26时，网格内x维度上有65536个块，而网格的x,
//y,z方向上的维度最大值是　　　65535　　　　　　，块的数量超出维度上限，所以会出现结果输出为零的情况．
//若需要处理更多的数据，那么网格的维度需要变为二维乃至三维才能正常处理．
//******************************************************************************************************************//

#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#define k1 1024
#define k2 1024
//检查宏，用于检测cuda编程出错位置及原因
#define CHECK(call)																\
{																				\
	const cudaError_t error=call;												\
	if(error!=cudaSuccess)														\
	{																			\
		printf("Error:%s:%d,",__FILE__,__LINE__);                               \
		printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));          \
		exit(1);		                                                        \
	}																			\
}																				\

using namespace std;

/*	＠property		核函数
　*	＠func			归约求每个共享内存内的最大值
　*	＠param_in		in		指向待归约的数据
　*	＠param_in		num		输入的数据量
　*	＠param_out		out		指向数据的输出地址
　*/
__global__ void reduce(int *in,int *out,int num)
{
	int tid=threadIdx.x;							//块内线程的索引
	int idx=blockIdx.x*blockDim.x+threadIdx.x;		//线程的实际索引
	
	extern __shared__  int data[];					//共享内存，空间大小在核函数调用时(func<<<,,共享内存字节数>>>)分配
	if(idx>=num)return;								//防止索引越界
	data[tid]=in[idx];								
	__syncthreads();								//等待共享内存拷贝数据结束

	for(unsigned int s=blockDim.x/2;s>0;s>>=1)		//块内归约操作
	{
		if(tid<s)
		{
			data[tid]=max(data[tid],data[tid+s]);
		}
		__syncthreads();
	}
	if(tid==0)out[blockIdx.x]=data[0];				//输出每个块内的归约结果
}



int main()
{	
	int  a;
	cout <<"数据量为2的几次方？(a<=25)"<<endl;
	cin>>a;
	
	int arraysize=1<<a;
	int arraybytes=arraysize*sizeof(int);
	int grid1,grid2;
	int *h_in,*h_cpu,*h_gpu,*d_in,*d_out,*d_tmp1,*d_tmp2;
	clock_t start,end;
	double time;
	h_in=(int *)malloc(arraybytes);
	h_cpu=(int *)malloc(sizeof(int));
	h_gpu=(int *)malloc(sizeof(int));
	
	cout <<"数据量: "<<arraysize<<endl;
	
	for(int i=0;i<arraysize;i++)				//生成数据
	{
		//h_in[i]=(int)random();
		h_in[i]=i;
	}	
	
	*h_cpu=0;
	start=clock();
	for(int i=0;i<arraysize;i++)				//CPU串行求最大值
	{
		*h_cpu=max(*h_cpu,h_in[i]);
	}
	end=clock();
	time=end-start;
	cout <<"cpu时间: "<<time/1000<<"ms"<<endl;
			
	grid1=(arraysize-1)/k1+1;					//设置网格大小(一维)
	cout <<"网格1大小："<<grid1 <<endl;
	grid2=(grid1-1)/k2+1;						//设置网格大小(一维)
	cout <<"网格2大小："<<grid2 <<endl;
	cudaMalloc((void **)&d_in,arraybytes);				//分配显存
	cudaMalloc((void **)&d_tmp1,grid1*sizeof(int));
	cudaMalloc((void **)&d_tmp2,grid2*sizeof(int));
	cudaMalloc((void **)&d_out,sizeof(int));	

/*	for(int i=0;i<arraysize;i++)
	{
		cout << h_in[i] <<"   ";
	}
	打印数据
	*/
	
	CHECK(cudaMemcpy(d_in,h_in,arraybytes,cudaMemcpyHostToDevice));
	reduce<<<grid1,k1,k1*sizeof(int)>>>(d_in,d_tmp1,arraysize);
//	CHECK(cudaDeviceSynchronize());				检查核函数运行错误，输出错误位置及错误信息
	reduce<<<grid2,k2,k2*sizeof(int)>>>(d_tmp1,d_tmp2,grid1);
//	CHECK(cudaDeviceSynchronize());	
	reduce<<<1,grid2,grid2*sizeof(int)>>>(d_tmp2,d_out,grid2);
//	CHECK(cudaDeviceSynchronize());	
	CHECK(cudaMemcpy(h_gpu,d_out,sizeof(int),cudaMemcpyDeviceToHost));
	
	cout <<"cpu归约结果:"<<*h_cpu<<endl;
	cout << "gpu归约结果:"<<*h_gpu <<endl;

	free(h_in);									//释放内存
	free(h_cpu);
	free(h_gpu);	
	cudaFree(d_in);								//释放显存
	cudaFree(d_tmp1);
	cudaFree(d_tmp2);
	cudaFree(d_out);
	return 0;
}
