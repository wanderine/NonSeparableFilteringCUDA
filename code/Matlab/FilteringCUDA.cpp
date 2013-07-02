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
    
	//-----------------------
	// Output pointers

	double     		*h_Filter_Response_double;
    float    		*h_Filter_Response;
	
    //---------------------

    /* Check the number of input and output arguments. */
    if(nrhs<2)
    {
        mexErrMsgTxt("Too few input arguments.");
    }
    if(nrhs>2)
    {
        mexErrMsgTxt("Too many input arguments.");
    }
    if(nlhs<1)
    {
        mexErrMsgTxt("Too few output arguments.");
    }
    if(nlhs>1)
    {
        mexErrMsgTxt("Too many output arguments.");
    }
   
    /* Input arguments */    

    // The data
    h_Data_double =  (double*)mxGetData(prhs[0]);
    h_Filter_double = (double*)mxGetData(prhs[1]);

	int NUMBER_OF_DIMENSIONS = mxGetNumberOfDimensions(prhs[0]);
	const int *ARRAY_DIMENSIONS_DATA = mxGetDimensions(prhs[0]);
    const int *ARRAY_DIMENSIONS_FILTER = mxGetDimensions(prhs[1]);
    
    int DATA_H, DATA_W, DATA_D, DATA_T;
    int FILTER_H, FILTER_W, FILTER_D, FILTER_T;
    
    if (NUMBER_OF_DIMENSIONS == 2)
    {
        DATA_H = ARRAY_DIMENSIONS_DATA[0];
        DATA_W = ARRAY_DIMENSIONS_DATA[1];
        DATA_D = 1;
        DATA_T = 1;
        
        FILTER_H = ARRAY_DIMENSIONS_FILTER[0];
        FILTER_W = ARRAY_DIMENSIONS_FILTER[1];
        FILTER_D = 1;
        FILTER_T = 1;        
    }
    else if (NUMBER_OF_DIMENSIONS == 3)
    {
        DATA_H = ARRAY_DIMENSIONS_DATA[0];
        DATA_W = ARRAY_DIMENSIONS_DATA[1];
        DATA_D = ARRAY_DIMENSIONS_DATA[2];
        DATA_T = 1;
        
        FILTER_H = ARRAY_DIMENSIONS_FILTER[0];
        FILTER_W = ARRAY_DIMENSIONS_FILTER[1];
        FILTER_D = ARRAY_DIMENSIONS_FILTER[2];
        FILTER_T = 1;        
    }
    else if (NUMBER_OF_DIMENSIONS == 4)
    {
        DATA_H = ARRAY_DIMENSIONS_DATA[0];
        DATA_W = ARRAY_DIMENSIONS_DATA[1];
        DATA_D = ARRAY_DIMENSIONS_DATA[2];
        DATA_T = ARRAY_DIMENSIONS_DATA[3];
        
        FILTER_H = ARRAY_DIMENSIONS_FILTER[0];
        FILTER_W = ARRAY_DIMENSIONS_FILTER[1];
        FILTER_D = ARRAY_DIMENSIONS_FILTER[2];
        FILTER_T = ARRAY_DIMENSIONS_FILTER[3];        
    }

    
    int DATA_SIZE = DATA_W * DATA_H * DATA_D * DATA_T * sizeof(float);
    int FILTER_SIZE = FILTER_W * FILTER_H * FILTER_D * FILTER_T * sizeof(float);

    if (NUMBER_OF_DIMENSIONS == 2)
    {
        mexPrintf("Data size : %i x %i \n",  DATA_W, DATA_H);
        mexPrintf("Filter size : %i x %i \n",  FILTER_W, FILTER_H);
    }    
    else if (NUMBER_OF_DIMENSIONS == 3)
    {
        mexPrintf("Data size : %i x %i x %i  \n",  DATA_W, DATA_H, DATA_D);
        mexPrintf("Filter size : %i x %i x %i \n",  FILTER_W, FILTER_H, FILTER_D);
    }    
    else if (NUMBER_OF_DIMENSIONS == 4)
    {
        mexPrintf("Data size : %i x %i x %i x %i \n",  DATA_W, DATA_H, DATA_D, DATA_T);
        mexPrintf("Filter size : %i x %i x %i x %i \n",  FILTER_W, FILTER_H, FILTER_D, FILTER_T);
    }
    
	//-------------------------------------------------
	// Output to Matlab

	// Create pointer for volumes to Matlab
	
    
	if (NUMBER_OF_DIMENSIONS == 2)
    {
        int ARRAY_DIMENSIONS_OUT[2];
        ARRAY_DIMENSIONS_OUT[0] = DATA_H;
        ARRAY_DIMENSIONS_OUT[1] = DATA_W;
        
        plhs[0] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT,mxDOUBLE_CLASS, mxREAL);
        h_Filter_Response_double = mxGetPr(plhs[0]); 	
    }    
    else if (NUMBER_OF_DIMENSIONS == 3)
    {
        int ARRAY_DIMENSIONS_OUT[3];
        ARRAY_DIMENSIONS_OUT[0] = DATA_H;
        ARRAY_DIMENSIONS_OUT[1] = DATA_W;
        ARRAY_DIMENSIONS_OUT[2] = DATA_D;
        
        plhs[0] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT,mxDOUBLE_CLASS, mxREAL);
        h_Filter_Response_double = mxGetPr(plhs[0]); 	
    }
    else if (NUMBER_OF_DIMENSIONS == 4)
    {
        int ARRAY_DIMENSIONS_OUT[4];
        ARRAY_DIMENSIONS_OUT[0] = DATA_H;
        ARRAY_DIMENSIONS_OUT[1] = DATA_W;
        ARRAY_DIMENSIONS_OUT[2] = DATA_D;
        ARRAY_DIMENSIONS_OUT[3] = DATA_T;
        
        plhs[0] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT,mxDOUBLE_CLASS, mxREAL);
        h_Filter_Response_double = mxGetPr(plhs[0]); 	
    }
        	    
	// ------------------------------------------------
 
	// Allocate memory on the host
	h_Data                         = (float *)mxMalloc(DATA_SIZE);    
    h_Filter                       = (float *)mxMalloc(FILTER_SIZE);
    h_Filter_Response              = (float *)mxMalloc(DATA_SIZE);
	
	// Reorder and cast data
	//pack_double2float_volumes(h_Data, h_Data_double, DATA_W, DATA_H, DATA_D, DATA_T);
    //pack_double2float_volumes(h_Filter, h_Filter_double, FILTER_W, FILTER_H, FILTER_D, FILTER_T);
	
    pack_double2float_image(h_Data, h_Data_double, DATA_W, DATA_H);
	pack_double2float_image(h_Filter, h_Filter_double, FILTER_W, FILTER_H);
	
    //------------------------
    
    Filtering my_Convolver(NUMBER_OF_DIMENSIONS, DATA_W, DATA_H, DATA_D, DATA_T, FILTER_W, FILTER_H, FILTER_D, FILTER_T, h_Data, h_Filter, h_Filter_Response, 1);
    //my_Convolver.DoConvolution2DTexture();
    my_Convolver.DoConvolution2DShared();
    //int error = my_Convolver.DoConvolution2DShared_();
    //mexPrintf("Error: %i \n", error);
    double convolution_time = my_Convolver.GetConvolutionTime();        
    mexPrintf("Convolution: %f milliseconds \n", convolution_time*1000);
             
    //------------------------
    
    //unpack_float2double_volumes(h_Filter_Response_double, h_Filter_Response, DATA_W, DATA_H, DATA_D, DATA_T);
    unpack_float2double_image(h_Filter_Response_double, h_Filter_Response, DATA_W, DATA_H);
    
    // Free all the allocated memory on the host
	mxFree(h_Data);
	mxFree(h_Filter);
    mxFree(h_Filter_Response);
    	
    //mexAtExit(cleanUp);
    
 	return;
}


