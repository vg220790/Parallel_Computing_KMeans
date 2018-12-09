#include "header.h"
#define MAX_THREADS_PER_BLOCK 1024

/*
For each point Pi, when previous position (xi, yi, zi) and velocity (vxi, vyi, vzi) are known
Its position at the given time 't' can be calculated as follows: xi(t) = xi + t*vxi , yi(t) = yi + t*vyi , zi(t) = zi + t*vzi
In this function each thread performs calculation for a certain point's new coordinate in the Kernel
*/
__global__ void calculateNewCoordinateKernel(int num_of_threads, double* dev_CurrentCoords, double* dev_InitCoords, double* dev_Velocities, double t)
{
	//int i = threadIdx.x;
	int threadId = threadIdx.x + (blockIdx.x * MAX_THREADS_PER_BLOCK);
	if (threadId < num_of_threads)
		dev_CurrentCoords[threadId] = dev_InitCoords[threadId] + (dev_Velocities[threadId] * t);
}

void error(double* dev_currentPointsCordinates, double* dev_pointsVelocityArr, double* dev_initPointsCordinates)
{
	cudaFree(dev_currentPointsCordinates);
	cudaFree(dev_pointsVelocityArr);
	cudaFree(dev_initPointsCordinates);
}

boolean calculatePointsCoordinatesWithCuda(double time, double* array_of_initial_points_coordinates, double* array_of_points_velocities, double* array_of_current_points_coordinates, int size)
{
	cudaError_t cudaStatus;
	cudaStatus = computeWithCuda(time, array_of_initial_points_coordinates, array_of_points_velocities, array_of_current_points_coordinates, size);

	//cudaDeviceReset must be called before exiting in order for profiling and
	//tracing tools such as Nsight and Visual Profiler to show complete traces
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!"); fflush(stdout);
		return FALSE;
	}
	return TRUE;
}

cudaError_t computeWithCuda(double time, double* array_of_initial_points_coordinates, double* array_of_points_velocities, double* array_of_current_points_coordinates, int size)
{
	cudaError_t cudaStatus;
	int parts;
	int blockAmount;
	double* dev_input_InitCoords = 0;
	double* dev_input_Velocities = 0;
	double* dev_output_CurrentCoords = 0;

	//size = numOfPointsPerProcess * 3
	//printf("\nin computePointsCoordinates function : size = %d", size);
	blockAmount = size / MAX_THREADS_PER_BLOCK;
	if (size % MAX_THREADS_PER_BLOCK != 0)
		blockAmount += (size % MAX_THREADS_PER_BLOCK);

	parts = size / 1000;
	if (size < 50000)
	{
		parts = size / 100;
	}

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?"); fflush(stdout);
		error(dev_output_CurrentCoords, dev_input_Velocities, dev_input_InitCoords);
	}

	// Allocate GPU buffers for three vectors (two input, one output)   
	cudaStatus = cudaMalloc((void**)&dev_output_CurrentCoords, size * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!"); fflush(stdout);
		error(dev_output_CurrentCoords, dev_input_Velocities, dev_input_InitCoords);
	}

	cudaStatus = cudaMalloc((void**)&dev_input_InitCoords, size * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!"); fflush(stdout);
		error(dev_output_CurrentCoords, dev_input_Velocities, dev_input_InitCoords);
	}

	cudaStatus = cudaMalloc((void**)&dev_input_Velocities, size * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!"); fflush(stdout);
		error(dev_output_CurrentCoords, dev_input_Velocities, dev_input_InitCoords);
	}

	//cudaStatus = cudaMemcpy(dev_output_CurrentCoords, array_of_current_points_coordinates, size * sizeof(double), cudaMemcpyHostToDevice);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMemcpy failed!"); fflush(stdout);
	//	error(dev_output_CurrentCoords, dev_input_Velocities, dev_input_InitCoords);
	//}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_input_InitCoords, array_of_initial_points_coordinates, size * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!"); fflush(stdout);
		error(dev_output_CurrentCoords, dev_input_Velocities, dev_input_InitCoords);
	}

	cudaStatus = cudaMemcpy(dev_input_Velocities, array_of_points_velocities, size * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!"); fflush(stdout);
		error(dev_output_CurrentCoords, dev_input_Velocities, dev_input_InitCoords);
	}

	// Launch a kernel on the GPU with 'parts' threads for each element.
	//calculateNewCoordinateKernel << <parts, size / parts >> >(dev_output_CurrentCoords, dev_input_InitCoords, dev_input_Velocities, time);
	calculateNewCoordinateKernel << <blockAmount, MAX_THREADS_PER_BLOCK >> >(size, dev_output_CurrentCoords, dev_input_InitCoords, dev_input_Velocities, time);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); fflush(stdout);
		error(dev_output_CurrentCoords, dev_input_Velocities, dev_input_InitCoords);
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		error(dev_output_CurrentCoords, dev_input_Velocities, dev_input_InitCoords);
	}

	//printf("Copy output vector from GPU buffer to host memory");
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(array_of_current_points_coordinates, dev_output_CurrentCoords, size * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		error(dev_output_CurrentCoords, dev_input_Velocities, dev_input_InitCoords);
	}
	return cudaStatus;
}



