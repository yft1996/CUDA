#include <cuda_runtime.h>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include "common.h"
#include "reduce.cuh"
#define  blocksize 512
using namespace cv;
__global__ void warmup(void){}
int recursiveReduce(int *data, int const size)
{
    // 数据量检查
    if (size == 1) return data[0];
    // 跨步定义
    int const stride = size / 2;
    // 归约循环
    for (int i = 0; i < stride; i++)
    {
        data[i] += data[i + stride];
    }
    // 返回指针地址及最大偏移地址
    return recursiveReduce(data, stride);
}
void cudafunc()
{
    warmup<<<1,1>>>();
    int size = 1 << 24;
    size_t bytes = size * sizeof(int);
    thrust::host_vector<int> h_vec1(size);
    int *tmp     = (int *) malloc(bytes);
    //生成数据
    for (int i = 0; i < size; i++)
    {
        h_vec1[i]=(int)( rand() & 0xFF );
    }
    double iStart, iElaps;//存储时间

    //host function
    memcpy (tmp, &h_vec1[0], bytes);
    iStart = seconds();
    int cpu_sum = recursiveReduce (tmp, size);
    iElaps = (seconds() - iStart)*1000;
    printf("cpu reduce      elapsed %f ms cpu_sum: %d\n", iElaps, cpu_sum);

    //基于完全展开和共享内存的求和操作
    int *d_idata = NULL;
    int *d_odata = NULL;
    dim3 block (blocksize, 1);
    dim3 grid  ((size + block.x - 1) / block.x, 1);
    int *h_odata = (int *) malloc(grid.x * sizeof(int));
    cudaMalloc((void **) &d_idata, bytes);
    cudaMalloc((void **) &d_odata, grid.x * sizeof(int));
    //h_vec1(100)为容器，使其参与显存拷贝操作需放入容器第一个元素的地址
    cudaMemcpy(d_idata, &h_vec1[0], bytes, cudaMemcpyHostToDevice);
    iStart = seconds();
    reduceCompleteUnroll<512><<<grid.x / 8, block>>>(d_idata, d_odata,size);
    cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int),cudaMemcpyDeviceToHost);
    iElaps = (seconds() - iStart)*1000;
    //当数据量足够大时可考虑将串行转为并行
    int gpu_sum = 0;
    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu Cmptnroll   elapsed %f ms gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);

    //基于thrust库的并行求和操作
    thrust::device_vector<int> d_vec1 = h_vec1;
    iStart = seconds();
    int sum = thrust::reduce(d_vec1.begin(), d_vec1.end(), 0, thrust::plus<int>());
    iElaps = (seconds() - iStart)*1000;
    printf("thrust reduce   elapsed %f ms sum: %d\n", iElaps, sum);
    free(tmp);
    free(h_odata);

    // free device memory
    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));

}