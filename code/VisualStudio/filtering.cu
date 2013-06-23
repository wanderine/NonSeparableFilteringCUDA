#include "filtering.h"
#include "cuda.h"
//#include <cutil_inline.h>
#include "filtering_kernel.cu"

#include <helper_functions.h>
#include <helper_cuda.h>

Filtering::Filtering(int ndim, int dw, int dh, int dd, int dt, int fw, int fh, int fd, int ft, float* input_data, float* output_data, float* filters, int nf)
{
	NDIM = ndim;
    DATA_W = dw;
    DATA_H = dh;
	DATA_D = dd;
	DATA_T = dt;
    
    FILTER_W = fw;
    FILTER_H = fh;
	FILTER_D = fd;
	FILTER_T = ft;

	h_Data = input_data;
	h_Filter_Responses = output_data;

	h_Filters = filters;
	NUMBER_OF_FILTERS = nf;
}

Filtering::~Filtering()
{
}



double Filtering::DoConvolution2DShared()
{        
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);

	checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    // Allocate memory for the filter response and the image
    float *d_Data, *d_Filter_Response;
	
	cudaMalloc((void **)&d_Data,  DATA_W * DATA_H * sizeof(float));
    cudaMalloc((void **)&d_Filter_Response,  DATA_W * DATA_H * sizeof(float));

	// Copy image to GPU
	cudaMemcpy(d_Data, h_Data, DATA_W * DATA_H * sizeof(float), cudaMemcpyHostToDevice);


	// Copy filter coefficients to constant memory
	cudaMemcpyToSymbol(c_Filter, h_Filters, FILTER_W * FILTER_H * sizeof(float), 0, cudaMemcpyHostToDevice);

    // 32 threads along x, 16 along y
	threadsInX = 32;
	threadsInY = 32;

    // Round up to get sufficient number of blocks
    blocksInX = (int)ceil((float)DATA_W / (float)threadsInX);
    blocksInY = (int)ceil((float)DATA_H / (float)threadsInY);

    dim3 dimGrid  = dim3(blocksInX, blocksInY, 1);
    dim3 dimBlock = dim3(threadsInX, threadsInY, 1);

    int xBlockDifference = 16;
    int yBlockDifference = 16;

    // Do 2D convolution	
	Convolution_2D<<<dimGrid, dimBlock>>>(d_Filter_Responses, d_Data, DATA_W, DATA_H, xBlockDifference, yBlockDifference);

	// Copy result to host
	cudaMemcpy(h_Filter_Responses, d_Filter_Responses, DATA_W * DATA_H * sizeof(float), cudaMemcpyDeviceToHost);

    // Free allocated memory
	cudaFree( d_Data );
    cudaFree( d_Filter_Response );

	checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    double gpuTime = 0.001 * sdkGetTimerValue(&hTimer);
    
	sdkDeleteTimer(&hTimer);

    cudaDeviceReset();
	return gpuTime;
}
