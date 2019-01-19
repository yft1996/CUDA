/*
 *	日期:	2019-1-19
 *	时间:	14:51
 *	姓名:	杨丰拓
 */
//***********************************************************************************************//
//	对大型矩阵进行转置操作时,读取操作为合并访问,写入操作为跨步访问.跨步的步长等于矩阵阶数(N),矩阵越大,
//步长越大.GPU每次以32个或128个元素为一组访问内存,合并访问时访问内存的次数最少,跨步访问时步长越大访问内
//存的次数越多,相应的时间耗费大大增加.所以为了优化程序,需要使写入操作又跨步访问优化为合并访问.
//优化措施即为使用共享内存作为中转站:
//输入矩阵[(块的行索引+线程的行索引)*N+(块的列索引+线程的列索引)]>>共享内存[线程的行索引*K+线程的列索引]
//输出时为块内元素的转置与块本身的转置
//共享内存[线程的列索引*K+线程的行索引]>>输入矩阵[(块的列索引+线程的行索引)*N+(块的行索引+线程的列索引)]
//为了实现合并访问全局内存,访问索引应始终为  线程的行索引*N+线程的列索引  (块的行列颠倒不会破坏合并访问)
//经检测K=16时,耗费的时间最少.
//************************************************************************************************//

#include <iostream>
#include <ctime>

using namespace std;
const int N= 1024;	
const int K= 16;	

/*  @property 	核函数
 *  @func     	使用共享内存(16*16)转置矩阵,将读合并写跨步优化为读写都合并
 *	@param_in 	in[]	输入矩阵
 *	@param_out	out[]	输出矩阵
 */
__global__ void transpose_parallel_per_element_tiled(float in[], float out[])
{
	int b_cols=blockIdx.x*blockDim.x;
	int b_rows=blockIdx.y*blockDim.y;
	int t_cols=threadIdx.x;
	int t_rows=threadIdx.y;
	__shared__ int data[K][K];
	data[t_rows][t_cols]=in[(b_rows+t_rows)*N+b_cols+t_cols];
	__syncthreads();

	out[(b_cols+t_rows)*N+(b_rows+t_cols)]=data[t_cols][t_rows];
}
__global__ void 
transpose_parallel_per_element(float in[], float out[])
{
	int i = blockIdx.x * K + threadIdx.x;
	int j = blockIdx.y * K + threadIdx.y;

	out[j + i*N] = in[i + j*N]; // out(j,i) = in(i,j)
}
/*  @property 	核函数
 *  @func     	预热GPU
 */
__global__ void warmup()
{	
	//空函数
}

/*  @property 	函数
 *  @func     	比较转置后的矩阵是否正确
 *	@param_in 	mat	指向待检测矩阵的指针
 *	@param_in	ref	指向参考矩阵的指针
 */
int compare_matrices(float *gpu, float *ref)
{
	int result = 0;

	for(int j=0; j < N; j++)
    	for(int i=0; i < N; i++)
    		if (ref[i + j*N] != gpu[i + j*N])
    		{
    			result = 1;
    		}
    return result;
}

/*  @property 	函数
 *  @func     	打印矩阵
 *	@param_in 	mat	指向待打印矩阵的指针
 */
void print_matrix(float *mat)
{
	for(int j=0; j < N; j++) 
	{
		for(int i=0; i < N; i++) {cout << mat[i + j*N]; }
		cout <<endl;
	}	
}
/*  @property 	函数
 *  @func     	使用CPU转置矩阵
 *	@param_in 	in[]	输入矩阵
 *	@param_out	out[]	输出矩阵
 */
void 
transpose_CPU(float in[], float out[])
{
	for(int j=0; j < N; j++)
    	for(int i=0; i < N; i++)
      		out[j + i*N] = in[i + j*N]; 
}

/*  @property 	函数
 *  @func     	构建一个N*N的矩阵
 *	@param_out 	mat	指向待填充矩阵的指针
 */
void fill_matrix(float *mat)
{
	for(int j=0; j < N * N; j++)
		mat[j] = (float) random();
}


int main()
{
	int numbytes = N * N * sizeof(float);


	clock_t start,end;
	double time1=0,time2=0,time3=0,time4=0,time=0;

for(int i=0;i<1000;i++)
{	
	float *in = (float *) malloc(numbytes);
	float *out = (float *) malloc(numbytes);
	float *gold = (float *) malloc(numbytes);
	float *d_in, *d_out;
	fill_matrix(in);
	start=clock();
	transpose_CPU(in, gold);
	end =clock();
	time+=end-start;

		
	start=clock();	
	warmup<<<1,1>>>();
	cudaDeviceSynchronize();	
	end =clock();
	time1+=end-start;

		
	start=clock();
	cudaMalloc(&d_in, numbytes);
	cudaMalloc(&d_out, numbytes);
	cudaMemcpy(d_in, in, numbytes, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	end=clock();
	time2+=end-start;

	
	dim3 blocks(N/K,N/K);
	dim3 threads(K,K);
	
	start=clock();
	transpose_parallel_per_element<<<blocks,threads>>>(d_in, d_out);
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	end=clock();
	time3+=end-start;	

/*	if( compare_matrices(out,gold))
	{
		cout <<"GPU转置失败"<<endl;
	}
	else 
	{
		cout <<"GPU转置成功"<<endl;
	}*/
		
	start=clock();
	transpose_parallel_per_element_tiled<<<blocks,threads>>>(d_in, d_out);
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	end=clock();
	time4+=end-start;	

/*	if( compare_matrices(out,gold))
	{
		cout <<"GPU2转置失败"<<endl;
	}
	else 
	{
		cout <<"GPU2转置成功"<<endl;
	}*/
	free(in);
	free(out);
	free(gold);
	cudaFree(d_in);
	cudaFree(d_out);
}
	cout <<"使用CPU转置时间为:" << time/1000 << "ms" <<endl;	
	cout <<"GPU预热时间为:" << time1/1000 << "ms" <<endl;
	cout <<"GPU分配内存时间为:" << time2/1000 << "ms" <<endl;
	cout <<"GPU转置(每线程一个元素)的时间为:" << time3/1000 <<"ms" <<endl;
	cout <<"GPU耗费总时间:" << (time3+time1+time2)/1000 << "ms" <<endl;
	cout <<"GPU转置(共享内存中转)的时间为:" << time4/1000 <<"ms" <<endl;
	cout <<"GPU2耗费总时间:" << (time4+time1+time2)/1000 << "ms" <<endl;

}
