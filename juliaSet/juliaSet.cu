#include <iostream>
#include <arrayfire.h>

#define DIMENSION 1000 // Number of pixels of the image

using namespace af;

__device__ int julia(int x, int y)
{
	cuFloatComplex constant;
	constant.x = -0.8;
	constant.y = 0.156;

	cuFloatComplex pt;
	
	float multiplier = 1.5; // Zoom factor

	// Calculate the point in complex plane corresponding to the x and y index
	float realPart = (multiplier * (float)(DIMENSION/2 - x) / (float)(DIMENSION/2));
	float complexPart = (multiplier * (float)(DIMENSION/2 - y) / (float)(DIMENSION/2));

	pt.x = realPart;
	pt.y = complexPart;

	for(int i = 0; i<200; i++)
	{
		pt = cuCmulf(pt,pt);
		pt = cuCaddf(pt,constant);    
		
		// Return 0 if point falls outside Julia set
		if(((pt.x*pt.x)+(pt.y*pt.y)) > 1000)		
			return 0;
	}

	// Return 1 if point falls inside Julia set
	return 1;
}

__global__ void checkJulia(int *result)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = (y_index * gridDim.x * blockDim.x ) + x_index;

	if(offset<(DIMENSION*DIMENSION))
	{
		// Sets result[offset] to 1 if point is in Julia set else sets it to 0
		result[offset] = julia(x_index, y_index);
	}
}

int main()
{
	int *h_result = (int*)malloc(sizeof(int) * DIMENSION * DIMENSION);
	int *d_result; // Holds 0 or 1 depending on weather the correspoding point is in Julia Set
	cudaMalloc((void**)&d_result, sizeof(int) * DIMENSION * DIMENSION);

	dim3 blockDim;
	blockDim.x = 8;
	blockDim.y = 8;

	dim3 gridDim;
	gridDim.x = DIMENSION/blockDim.x;
	gridDim.y = DIMENSION/blockDim.y;

	if(DIMENSION%blockDim.x != 0)
		gridDim.x++;

	if(DIMENSION%blockDim.y != 0)
		gridDim.y++;

	// Check for each pixel if it is in the Julia Set
	checkJulia<<<gridDim,blockDim>>>(d_result);
	cudaDeviceSynchronize();
	cudaMemcpy(h_result, d_result, sizeof(int) * DIMENSION * DIMENSION, cudaMemcpyDeviceToHost);
	cudaFree(d_result);
	array img(DIMENSION, DIMENSION, 3);

	// Sets the point to blue color if point is in Julia Set, else sets it to black
	for(int i = 0; i<DIMENSION; i++)
	{
		for(int j = 0; j<DIMENSION; j++)
		{
			int offset = i*DIMENSION+j;
			img(i, j, 0) = 0;
			img(i, j, 1) = 0;
			img(i, j, 2) = 255 * h_result[offset];
		}
	}
	image(img);

	std::cin.get();
	return 0;
}
