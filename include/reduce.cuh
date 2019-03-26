#ifndef __REDUCE_CUH__
#define __REDUCE_CUH__
/*
 * 求和调用示例
 * reduceCompleteUnroll<512><<<grid.x / 8, block>>>(d_idata, d_odata,size);
 */
template <unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(int *g_idata, int *g_odata,unsigned int n);
/*
 * 求最大值调用示例
 * reduce_max<512><<<grid.x / 8, block>>>(d_idata, d_odata,size);
 */
template <unsigned int iBlockSize>
__global__ void reduce_max(int *g_idata,int *g_odata,unsigned int n);
/*
 * 求最小值调用示例
 * reduce_min<512><<<grid.x / 8, block>>>(d_idata, d_odata,size);
 */
template <unsigned int iBlockSize>
__global__ void reduce_min(int *g_idata,int *g_odata,unsigned int n);

//***************************************************************************//

/***
 * @tparam  iBlockSize 　线程块大小
 * @param   g_idata     输入指针
 * @param   g_odata     输出指针
 * @param   n           最大偏移地址
 * @return
 */
//归约模板
template <unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(int *g_idata, int *g_odata,unsigned int n)
{

    unsigned int tid = threadIdx.x;//块内索引
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;//全局索引（展开循环８次）

    // 将全局数据指针转换为此块的本地指针
    //int *idata = g_idata + blockIdx.x * blockDim.x * 8;
    //使用共享内存
    __shared__ int s[iBlockSize];
    // 读取８个线程块大小的数据量
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        s[tid] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }
    __syncthreads();

    // 完全展开for循环,线程块＝＝512时，从第二个判断语句开始执行
    if (iBlockSize >= 1024 && tid < 512) s[tid] += s[tid + 512];
    __syncthreads();

    if (iBlockSize >= 512 && tid < 256)  s[tid] += s[tid + 256];
    __syncthreads();

    if (iBlockSize >= 256 && tid < 128)  s[tid] += s[tid + 128];
    __syncthreads();

    if (iBlockSize >= 128 && tid < 64)   s[tid] += s[tid + 64];
    __syncthreads();

    // 展开线程束
    if (tid < 32)
    {
        volatile int *vsmem = s;//volatile 关键字起屏障作用
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }
    // 结果写回全局内存
    if (tid == 0) g_odata[blockIdx.x] = s[0];
}

template <unsigned int iBlockSize>
__global__ void reduce_max(int *g_idata,int *g_odata,unsigned int n)
{
    unsigned int tid = threadIdx.x;//块内索引
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;//全局索引（展开循环８次）

    // 将全局数据指针转换为此块的本地指针
    //int *idata = g_idata + blockIdx.x * blockDim.x * 8;
    //使用共享内存
    __shared__ int idata[iBlockSize];
    // 读取８个线程块大小的数据量
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        idata[tid] = max(a1,max(a2,max(a3,max(a4,max(b1,max(b2,max(b3,b4)))))));
    }
    __syncthreads();

    // 完全展开for循环,线程块＝＝512时，从第二个判断语句开始执行
    if (iBlockSize >= 1024 && tid < 512) idata[tid] =max(idata[tid],idata[tid + 512]);
    __syncthreads();

    if (iBlockSize >= 512 && tid < 256)  idata[tid] =max(idata[tid],idata[tid + 256]);
    __syncthreads();

    if (iBlockSize >= 256 && tid < 128)  idata[tid] =max(idata[tid],idata[tid + 128]);
    __syncthreads();

    if (iBlockSize >= 128 && tid < 64)   idata[tid] =max(idata[tid],idata[tid + 64]);
    __syncthreads();

    // 展开线程束
    if (tid < 32)
    {
        volatile int *vsmem = idata;//volatile 关键字起屏障作用
        vsmem[tid] =max(vsmem[tid],vsmem[tid + 32]);
        vsmem[tid] =max(vsmem[tid],vsmem[tid + 16]);
        vsmem[tid] =max(vsmem[tid],vsmem[tid +  8]);
        vsmem[tid] =max(vsmem[tid],vsmem[tid +  4]);
        vsmem[tid] =max(vsmem[tid],vsmem[tid +  2]);
        vsmem[tid] =max(vsmem[tid],vsmem[tid +  1]);
    }
    // 结果写回全局内存
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

template <unsigned int iBlockSize>
__global__ void reduce_min(int *g_idata,int *g_odata,unsigned int n)
{
    unsigned int tid = threadIdx.x;//块内索引
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;//全局索引（展开循环８次）

    // 将全局数据指针转换为此块的本地指针
    //int *idata = g_idata + blockIdx.x * blockDim.x * 8;
    //使用共享内存
    __shared__ int idata[iBlockSize];
    // 读取８个线程块大小的数据量
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        idata[tid] = min(a1,min(a2,min(a3,min(a4,min(b1,min(b2,min(b3,b4)))))));
    }
    __syncthreads();

    // 完全展开for循环,线程块＝＝512时，从第二个判断语句开始执行
    if (iBlockSize >= 1024 && tid < 512) idata[tid] =min(idata[tid],idata[tid + 512]);
    __syncthreads();

    if (iBlockSize >= 512 && tid < 256)  idata[tid] =min(idata[tid],idata[tid + 256]);
    __syncthreads();

    if (iBlockSize >= 256 && tid < 128)  idata[tid] =min(idata[tid],idata[tid + 128]);
    __syncthreads();

    if (iBlockSize >= 128 && tid < 64)   idata[tid] =min(idata[tid],idata[tid + 64]);
    __syncthreads();

    // 展开线程束
    if (tid < 32)
    {
        volatile int *vsmem = idata;//volatile 关键字起屏障作用
        vsmem[tid] =min(vsmem[tid],vsmem[tid + 32]);
        vsmem[tid] =min(vsmem[tid],vsmem[tid + 16]);
        vsmem[tid] =min(vsmem[tid],vsmem[tid +  8]);
        vsmem[tid] =min(vsmem[tid],vsmem[tid +  4]);
        vsmem[tid] =min(vsmem[tid],vsmem[tid +  2]);
        vsmem[tid] =min(vsmem[tid],vsmem[tid +  1]);
    }
    // 结果写回全局内存
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}
#endif //__REDUCE_CUH__