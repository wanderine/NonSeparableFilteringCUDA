#include "mex.h"
#include "help_functions.cpp"
#include "filtering.h"
#include <cuda.h>
#include <cuda_runtime.h>

void cleanUp()
{
    cudaDeviceReset();
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    //-----------------------
    // Input pointers
    
    double		    *h_Data_double;
    float           *h_Data;
    
    double	        *h_Filter_double;
    float		    *h_Filter;
    
    int             TIMING;
    
    //-----------------------
    // Output pointers
    
    double     		*h_Filter_Response_Texture_double;
    double     		*h_Filter_Response_Texture_Unrolled_double;
    double     		*h_Filter_Response_Shared_double;
    double     		*h_Filter_Response_Shared_Unrolled_double;
    float    		*h_Filter_Response;
    
    double          *convolution_time_texture, *convolution_time_texture_unrolled, *convolution_time_shared, *convolution_time_shared_unrolled, *convolution_time_fft;
    
    //---------------------
    
    /* Check the number of input and output arguments. */
    if(nrhs<3)
    {
        mexErrMsgTxt("Too few input arguments.");
    }
    if(nrhs>3)
    {
        mexErrMsgTxt("Too many input arguments.");
    }
    if(nlhs<9)
    {
        mexErrMsgTxt("Too few output arguments.");
    }
    if(nlhs>9)
    {
        mexErrMsgTxt("Too many output arguments.");
    }
    
    /* Input arguments */
    
    // The data
    h_Data_double =  (double*)mxGetData(prhs[0]);
    h_Filter_double = (double*)mxGetData(prhs[1]);
    TIMING = (int)mxGetScalar(prhs[2]);
    
    int NUMBER_OF_DIMENSIONS = mxGetNumberOfDimensions(prhs[0]);
    const int *ARRAY_DIMENSIONS_DATA = mxGetDimensions(prhs[0]);
    const int *ARRAY_DIMENSIONS_FILTER = mxGetDimensions(prhs[1]);
    
    int DATA_H, DATA_W;
    int FILTER_H, FILTER_W;
    
    DATA_H = ARRAY_DIMENSIONS_DATA[0];
    DATA_W = ARRAY_DIMENSIONS_DATA[1];
    
    FILTER_H = ARRAY_DIMENSIONS_FILTER[0];
    FILTER_W = ARRAY_DIMENSIONS_FILTER[1];
    
    int DATA_SIZE = DATA_W * DATA_H * sizeof(float);
    int FILTER_SIZE = FILTER_W * FILTER_H * sizeof(float);
    
    mexPrintf("Data size : %i x %i \n",  DATA_W, DATA_H);
    mexPrintf("Filter size : %i x %i \n",  FILTER_W, FILTER_H);
    
    //-------------------------------------------------
    // Output to Matlab
    
    // Create pointer for volumes to Matlab
    
    int ARRAY_DIMENSIONS_OUT[2];
    ARRAY_DIMENSIONS_OUT[0] = DATA_H;
    ARRAY_DIMENSIONS_OUT[1] = DATA_W;
    
    plhs[0] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT,mxDOUBLE_CLASS, mxREAL);
    h_Filter_Response_Texture_double = mxGetPr(plhs[0]);
    
    plhs[1] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT,mxDOUBLE_CLASS, mxREAL);
    h_Filter_Response_Texture_Unrolled_double = mxGetPr(plhs[1]);
    
    plhs[2] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT,mxDOUBLE_CLASS, mxREAL);
    h_Filter_Response_Shared_double = mxGetPr(plhs[2]);
    
    plhs[3] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT,mxDOUBLE_CLASS, mxREAL);
    h_Filter_Response_Shared_Unrolled_double = mxGetPr(plhs[3]);
    
    //---------
    
    plhs[4] = mxCreateDoubleMatrix(1, 1, mxREAL);
    convolution_time_texture = mxGetPr(plhs[4]);
    
    plhs[5] = mxCreateDoubleMatrix(1, 1, mxREAL);
    convolution_time_texture_unrolled = mxGetPr(plhs[5]);
    
    plhs[6] = mxCreateDoubleMatrix(1, 1, mxREAL);
    convolution_time_shared = mxGetPr(plhs[6]);
    
    plhs[7] = mxCreateDoubleMatrix(1, 1, mxREAL);
    convolution_time_shared_unrolled = mxGetPr(plhs[7]);
    
    plhs[8] = mxCreateDoubleMatrix(1, 1, mxREAL);
    convolution_time_fft = mxGetPr(plhs[8]);
    
    
    // ------------------------------------------------
    
    // Allocate memory on the host
    h_Data                         = (float *)mxMalloc(DATA_SIZE);
    h_Filter                       = (float *)mxMalloc(FILTER_SIZE);
    h_Filter_Response              = (float *)mxMalloc(DATA_SIZE);
    
    // Reorder and cast data
    pack_double2float_image(h_Data, h_Data_double, DATA_W, DATA_H);
    pack_double2float_image(h_Filter, h_Filter_double, FILTER_W, FILTER_H);
    
    //------------------------
    
    Filtering my_Convolver(DATA_W, DATA_H, FILTER_W, FILTER_H, h_Data, h_Filter, h_Filter_Response);
    
    *convolution_time_texture = 0.0;
    *convolution_time_texture_unrolled = 0.0;
    *convolution_time_shared = 0.0;
    *convolution_time_shared_unrolled = 0.0;
    *convolution_time_fft = 0.0;
        
    int RUNS = 20;
    
    if (TIMING == 1)
    {
        my_Convolver.SetUnrolled(false);
        for (int i = 0; i < RUNS; i++)
        {
            my_Convolver.DoConvolution2DTexture();
            *convolution_time_texture += my_Convolver.GetConvolutionTime();
        }
        *convolution_time_texture /= (double)RUNS;
        
        my_Convolver.SetUnrolled(true);
        for (int i = 0; i < RUNS; i++)
        {
            my_Convolver.DoConvolution2DTexture();
            *convolution_time_texture_unrolled += my_Convolver.GetConvolutionTime();
        }
        *convolution_time_texture_unrolled /= (double)RUNS;
        
        my_Convolver.SetUnrolled(false);
        for (int i = 0; i < RUNS; i++)
        {
            my_Convolver.DoConvolution2DShared();
            *convolution_time_shared += my_Convolver.GetConvolutionTime();
        }
        *convolution_time_shared /= (double)RUNS;
        
        my_Convolver.SetUnrolled(true);
        for (int i = 0; i < RUNS; i++)
        {
            my_Convolver.DoConvolution2DShared();
            *convolution_time_shared_unrolled += my_Convolver.GetConvolutionTime();
        }
        *convolution_time_shared_unrolled /= (double)RUNS;
        
        for (int i = 0; i < RUNS; i++)
        {
            my_Convolver.DoFiltering2DFFT();
            *convolution_time_fft += my_Convolver.GetConvolutionTime();
        }
        *convolution_time_fft /= (double)RUNS;
    }
    
    
    //-------------------
    
    /*
    my_Convolver.SetUnrolled(false);
    my_Convolver.DoConvolution2DTexture();
    unpack_float2double_image(h_Filter_Response_Texture_double, h_Filter_Response, DATA_W, DATA_H);
    my_Convolver.SetUnrolled(true);
    my_Convolver.DoConvolution2DTexture();
    unpack_float2double_image(h_Filter_Response_Texture_Unrolled_double, h_Filter_Response, DATA_W, DATA_H);
    */
    my_Convolver.SetUnrolled(false);
    my_Convolver.DoConvolution2DShared();
    unpack_float2double_image(h_Filter_Response_Shared_double, h_Filter_Response, DATA_W, DATA_H);
    /*
    my_Convolver.SetUnrolled(true);
    my_Convolver.DoConvolution2DShared();
    unpack_float2double_image(h_Filter_Response_Shared_Unrolled_double, h_Filter_Response, DATA_W, DATA_H);
    */
    
    // Free all the allocated memory on the host
    mxFree(h_Data);
    mxFree(h_Filter);
    mxFree(h_Filter_Response);
    
    mexAtExit(cleanUp);
    
    return;
}


