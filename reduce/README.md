日期：2019-3-26

时间：10:42

姓名：杨丰拓

*.cu文件

	compare_reduce_add.cu
	介绍：对CPU下并行求和，GPU下并行求和以及thrust库函数求和的运行时间进行了对比
	备注：thrust容器与指针的相互转化
		h_vector<int> h(100);
		int *d=NULL;
		cudaMalloc(&d,100*sizeof(int));
		cudaMemcpy(d,&h[0],100*sizeof(100),cudaMemcpyHostToDevice);
