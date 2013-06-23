

#define VALID_FILTER_RESPONSES_X 48
#define VALID_FILTER_RESPONSES_Y 48

#ifndef FILTERING_H_
#define FILTERING_H_

class Filtering
{

public:

    Filtering(int ndim, int dw, int dh, int dd, int dt, int fw, int fh, int fd, int ft, float* input_data, float* output_data, float* filters, int nf);
    ~Filtering();

    double DoConvolution2DShared();
	double DoConvolution2DTexture();

	double DoConvolution3DShared();
	double DoConvolution3DTexture();
                  
	double DoConvolution4DShared();

private:


    //void CopyMonomialFilters(float* h_Monomial_Filter_1, float* h_Monomial_Filter_2, float* h_Monomial_Filter_3, float* h_Monomial_Filter_4, float* h_Monomial_Filter_5, float* h_Monomial_Filter_6, float* h_Monomial_Filter_7, float* h_Monomial_Filter_8, float* h_Monomial_Filter_9, float* h_Monomial_Filter_10, float* h_Monomial_Filter_11, float* h_Monomial_Filter_12, float* h_Monomial_Filter_13, float* h_Monomial_Filter_14, int z, int t, int FILTER_W, int FILTER_H, int FILTER_D);
    //void CopyDenoisingFilters(float* h_Denoising_Filter_1, float* h_Denoising_Filter_2, float* h_Denoising_Filter_3, float* h_Denoising_Filter_4, float* h_Denoising_Filter_5, float* h_Denoising_Filter_6, float* h_Denoising_Filter_7, float* h_Denoising_Filter_8, float* h_Denoising_Filter_9, float* h_Denoising_Filter_10, float* h_Denoising_Filter_11, int z, int t, int FILTER_W, int FILTER_H, int FILTER_D);

	// Number of dimensions
	int NDIM;

    // Data sizes	
	int DATA_W;
    int DATA_H;
	int DATA_D;
	int DATA_T;

	// Filter sizes
	int FILTER_W;
	int FILTER_H;
	int FILTER_D;
	int FILTER_T;

	int NUMBER_OF_FILTERS;

	// Host pointers
	float	*h_Data;
	float	*h_Filter_Responses;
	float	*h_Filters;
	
	// Device pointers
	float	*d_Data;
	float	*d_Filter_Responses;

	int threadsInX, threadsInY;
    int blocksInX, blocksInY;


};

#endif 