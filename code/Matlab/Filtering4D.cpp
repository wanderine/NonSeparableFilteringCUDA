/*
	Non-separable 2D, 3D and 4D Filtering with CUDA
	Copyright (C) <2013>  Anders Eklund, andek034@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

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
    
    double     		*h_Filter_Response_Shared_double;
    double     		*h_Filter_Response_Shared_Unrolled_double;
    float    		*h_Filter_Response;
    
    double          *convolution_time_shared, *convolution_time_shared_unrolled, *convolution_time_fft;
    
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
    if(nlhs<5)
    {
        mexErrMsgTxt("Too few output arguments.");
    }
    if(nlhs>5)
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
    
    int DATA_H, DATA_W, DATA_D, DATA_T;
    int FILTER_H, FILTER_W, FILTER_D, FILTER_T;
    
    DATA_H = ARRAY_DIMENSIONS_DATA[0];
    DATA_W = ARRAY_DIMENSIONS_DATA[1];
    DATA_D = ARRAY_DIMENSIONS_DATA[2];
    DATA_T = ARRAY_DIMENSIONS_DATA[3];
    
    FILTER_H = ARRAY_DIMENSIONS_FILTER[0];
    FILTER_W = ARRAY_DIMENSIONS_FILTER[1];
    FILTER_D = ARRAY_DIMENSIONS_FILTER[2];
    FILTER_T = ARRAY_DIMENSIONS_FILTER[3];
    
    int DATA_SIZE = DATA_W * DATA_H * DATA_D * DATA_T * sizeof(float);
    int FILTER_SIZE = FILTER_W * FILTER_H * FILTER_D * FILTER_T * sizeof(float);
    
    mexPrintf("Data size : %i x %i x %i x %i \n",  DATA_W, DATA_H, DATA_D, DATA_T);
    mexPrintf("Filter size : %i x %i x %i x %i\n",  FILTER_W, FILTER_H, FILTER_D, FILTER_T);
    
    //-------------------------------------------------
    // Output to Matlab
    
    // Create pointer for volumes to Matlab
    
    int ARRAY_DIMENSIONS_OUT[4];
    ARRAY_DIMENSIONS_OUT[0] = DATA_H;
    ARRAY_DIMENSIONS_OUT[1] = DATA_W;
    ARRAY_DIMENSIONS_OUT[2] = DATA_D;
    ARRAY_DIMENSIONS_OUT[3] = DATA_T;
       
    plhs[0] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT,mxDOUBLE_CLASS, mxREAL);
    h_Filter_Response_Shared_double = mxGetPr(plhs[0]);
    
    plhs[1] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT,mxDOUBLE_CLASS, mxREAL);
    h_Filter_Response_Shared_Unrolled_double = mxGetPr(plhs[1]);
    
    //---------
        
    plhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);
    convolution_time_shared = mxGetPr(plhs[2]);
    
    plhs[3] = mxCreateDoubleMatrix(1, 1, mxREAL);
    convolution_time_shared_unrolled = mxGetPr(plhs[3]);
    
    plhs[4] = mxCreateDoubleMatrix(1, 1, mxREAL);
    convolution_time_fft = mxGetPr(plhs[4]);
    
    
    // ------------------------------------------------
    
    // Allocate memory on the host
    h_Data                         = (float *)mxMalloc(DATA_SIZE);
    h_Filter                       = (float *)mxMalloc(FILTER_SIZE);
    h_Filter_Response              = (float *)mxMalloc(DATA_SIZE);
    
    // Reorder and cast data
    pack_double2float_volumes(h_Data, h_Data_double, DATA_W, DATA_H, DATA_D, DATA_T);
    pack_double2float_volumes(h_Filter, h_Filter_double, FILTER_W, FILTER_H, FILTER_D, FILTER_T);
    
    //------------------------
    
    Filtering my_Convolver(DATA_W, DATA_H, DATA_D, DATA_T, FILTER_W, FILTER_H, FILTER_D, FILTER_T, h_Data, h_Filter, h_Filter_Response);
    
    *convolution_time_shared = 0.0;
    *convolution_time_shared_unrolled = 0.0;
    *convolution_time_fft = 0.0;
    
    int RUNS = 10;
    
    if (TIMING == 1)
    {        
        my_Convolver.SetUnrolled(false);
        for (int i = 0; i < RUNS; i++)
        {
            my_Convolver.DoConvolution4DShared();
            *convolution_time_shared += my_Convolver.GetConvolutionTime();
        }
        *convolution_time_shared /= (double)RUNS;
        
        my_Convolver.SetUnrolled(true);
        for (int i = 0; i < RUNS; i++)
        {
            my_Convolver.DoConvolution4DShared();
            *convolution_time_shared_unrolled += my_Convolver.GetConvolutionTime();
        }
        *convolution_time_shared_unrolled /= (double)RUNS;
        
        for (int i = 0; i < RUNS; i++)
        {
            my_Convolver.DoFiltering4DFFT();
            *convolution_time_fft += my_Convolver.GetConvolutionTime();
        }
        *convolution_time_fft /= (double)RUNS;
    }
    
    //-------------------
    
    my_Convolver.SetUnrolled(false);
    my_Convolver.DoConvolution4DShared();
    *convolution_time_shared = my_Convolver.GetConvolutionTime();
    unpack_float2double_volumes(h_Filter_Response_Shared_double, h_Filter_Response, DATA_W, DATA_H, DATA_D, DATA_T);
    my_Convolver.SetUnrolled(true);
    my_Convolver.DoConvolution4DShared();
    *convolution_time_shared_unrolled = my_Convolver.GetConvolutionTime();
    unpack_float2double_volumes(h_Filter_Response_Shared_Unrolled_double, h_Filter_Response, DATA_W, DATA_H, DATA_D, DATA_T);
    
    my_Convolver.DoFiltering4DFFT();
    *convolution_time_fft = my_Convolver.GetConvolutionTime();
    
    
    
    
    // Free all the allocated memory on the host
    mxFree(h_Data);
    mxFree(h_Filter);
    mxFree(h_Filter_Response);
    
    mexAtExit(cleanUp);
    
    return;
}


