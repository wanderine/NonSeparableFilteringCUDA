#ifndef HELP_FUNCTIONS_CU_
#define HELP_FUNCTIONS_CU_



#define Complex float2



__device__ int Calculate_2D_Index(int x, int y, int DATA_W)
{
    return x + y * DATA_W;
}


__device__ int Calculate_3D_Index(int x, int y, int z, int DATA_W, int DATA_H)
{
    return x + y * DATA_W + z * DATA_W * DATA_H;
}


__device__ int Calculate_4D_Index(int x, int y, int z, int t, int DATA_W, int DATA_H, int DATA_D)
{
    return x + y * DATA_W + z * DATA_W * DATA_H + t * DATA_W * DATA_H * DATA_D;
}

__device__ int Calculate_4D_Index_Time_Circular(int x, int y, int z, int t, int DATA_W, int DATA_H, int DATA_D, int DATA_T)
{
	int tt;

	// Circular convolution in time
	if ((t < 0))
	{	
		tt = DATA_T + t;
	}
	else if (t >= DATA_T)
	{
		tt = t - DATA_T;
	}

	return x + y * DATA_W + z * DATA_W * DATA_H + tt * DATA_W * DATA_H * DATA_D;
}

__global__ void CopyData(float *Data1, float* Data2, int t, int DATA_W, int DATA_H, int DATA_D)
{   
	volatile int x = blockIdx.x * blockDim.x + threadIdx.x; 
	volatile int y = blockIdx.y * blockDim.y + threadIdx.y;  
	volatile int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
    if ( (x >= DATA_W) || (y >= DATA_H) || (z >= DATA_D) )
	    return;

	int idx = Calculate_4D_Index(x, y, z, t, DATA_W, DATA_H, DATA_D);

	Data1[idx] = Data2[idx];
}

__global__ void ResetVolumes(float* reset_volumes, const int DATA_W, const int DATA_H, const int DATA_D, const int DATA_T)
{
	volatile int x = blockIdx.x * blockDim.x + threadIdx.x;
    volatile int y = blockIdx.y * blockDim.y + threadIdx.y;
    volatile int z = blockIdx.z * blockDim.z + threadIdx.z;
	
	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	for (int t = 0; t < DATA_T; t++)
	{
		reset_volumes[Calculate_4D_Index(x, y, z, t, DATA_W, DATA_H, DATA_D)] = 0.0f;
	}
}



__device__ void ComplexMultAndScale(Complex a, Complex b, Complex& c, float constant)
{
    c.x = constant * (a.x * b.x - a.y * b.y);
	c.y = constant * (a.y * b.x + a.x * b.y);
}

__device__ float Angle(Complex& complex_number)
{
    return atan2f(complex_number.y, complex_number.x);
}

__device__ float Abs(Complex& complex_number)
{
    return sqrtf( complex_number.x * complex_number.x + complex_number.y * complex_number.y);
}







__global__ void ResetSlice(float* reset_slice, int DATA_W, int DATA_H)
{
	volatile int x = blockIdx.x * blockDim.x + threadIdx.x;
	volatile int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= DATA_W || y >= DATA_H)
		return;

	reset_slice[Calculate_2D_Index(x, y, DATA_W)] = 0.0f;
}

__global__ void SetSlices(float* reset_slices, int DATA_W, int DATA_H, int DATA_D)
{
	volatile int x = blockIdx.x * blockDim.x + threadIdx.x;
	volatile int y = blockIdx.y * blockDim.y + threadIdx.y;
	volatile int z = blockIdx.z * blockDim.z + threadIdx.z;	

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	reset_slices[Calculate_3D_Index(x, y, z, DATA_W, DATA_H)] = 1000.0f;
}

__global__ void ResetSlices(float* reset_slices, int DATA_W, int DATA_H, int DATA_D)
{
	volatile int x = blockIdx.x * blockDim.x + threadIdx.x;
	volatile int y = blockIdx.y * blockDim.y + threadIdx.y;
	volatile int z = blockIdx.z * blockDim.z + threadIdx.z;	

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	reset_slices[Calculate_3D_Index(x, y, z, DATA_W, DATA_H)] = 0.0f;
}


#endif
