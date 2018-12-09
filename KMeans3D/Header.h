#pragma once
#pragma once
#pragma once
#pragma once
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define DIM 3  //because ( x , y , z ) is 3 dimentional 
#define MASTER 0
#define TRANSFER_TAG 0
#define MID_TERMINATION_TAG 1
#define FINAL_TERMINATION_TAG 2
#define _CRT_SECURE_NO_WARNINGS
#define POINT_STRUCT_SIZE 8 
#define CLUSTER_STRUCT_SIZE 4 
#define INPUT_PATH "C:\\Users\\afeka\\Desktop\\KMeans3D\\KMeans3D\\input.txt"
#define OUTPUT_PATH "C:\\Users\\afeka\\Desktop\\KMeans3D\\KMeans3D\\output.txt"

typedef enum boolean { FALSE, TRUE };

//Defining Structs

typedef struct Cluster
{
	double x;
	double y;
	double z;
	double diameter; /* defined as largest distance between 2 points in a cluster */
};

typedef struct Point
{
	double x;
	double y;
	double z;
	double vx;
	double vy;
	double vz;
	int currentClusterIndex;
	int previousClusterIndex;
};


void init_new_MPI_Types(MPI_Datatype* MPI_Point, MPI_Datatype* MPI_Cluster);
Point* readDataFromFile(int* N, int* K, int* T, double* dt, double* LIMIT, double* QM);
void writeToFile(double t, double q, Cluster* clusters, int K);
void checkAllocation(void* pointer);
Cluster* getInitClusters(const Point* points, int K);
int getClosestClusterIndex(double x, double y, double z, Cluster* clusters, int K);
void assignPointsToClusters(Point** pointsMat, int* clustersSize, Point* points, int numOfPoints, Cluster* clusters, int K);
double calcNewDistance(double x1, double y1, double z1, double x2, double y2, double z2);
void calcNewClusterCenter(Cluster* cluster, Point* clusterPoints, int clusterPointsSize);
double getEvaluatedQuality(Point** pointsMat, Cluster* clusters, int K, int* clustersSize);
double calcNewClusterDiameter(Point* clusterPoints, int clusterPointsSize);
void calcPointsCoordinatesWithoutCuda(Point* points, int numPointsPerProcess, double t);
double runKMeansAlgorithm(Point* points, Cluster* clusters, Point** pointsMat, int* clustersSize, int numOfPoints, int K,
	int limit, double QM, double T, double dt, MPI_Datatype PointMPIType, MPI_Datatype ClusterMPIType, double* sumsArr,
	int numprocs, double* time, double* currentPointsCordinates, double* initialPointsCordinates, double* pointsVelocityArr);
double kMeansAlgorithmMaster(Point* points, Cluster* clusters, Point** pointsMat, int* clustersSize, int numOfPoints, int K,
	int limit, int numprocs, MPI_Datatype PointMPIType, MPI_Datatype ClusterMPIType, double* sumsArr);
void kMeansAlgorithmSlave(Point* points, Cluster* clusters, Point** pointsMat, int* clustersSize, int numOfPoints, int K
	, MPI_Datatype PointMPIType, MPI_Datatype ClusterMPIType, double* sumsArr);
void returnAllPointsToMaster(Point** pointsMat, int* clustersSize, int* totalClustersSize, int K, int numOfProcs, MPI_Datatype PointMPIType);
void getPointsFromMaster(int K, int* clustersSize, Point** pointsMat, MPI_Datatype PointMPIType);
void slaveDoWork(Point* points, Cluster* clusters, Point** pointsMat, int* clustersSize, int numOfPoints, int K,
	MPI_Datatype PointMPIType, MPI_Datatype ClusterMPIType, double* sumsArr, double* currentPointsCordinates, double * initialPointsCordinates, double* pointsVelocityArr);
void initCoordVelocArrays(double* currentPointsCordinates, double* initialPointsCordinates, double* pointsVelocityArr, Point* points, int numOfPoints);
void updateCoordinatesInPointsArray(Point* points, int numOfPoints, double* currentPointsCordinates);

//CUDA Kernel Functions 
__global__ void calculateNewCoordinateKernel(int num_of_threads, double* dev_initPointsCordinates, double* dev_pointsVelocityArr, double* dev_currentPointsCordinates, double time);
void error(double* dev_currentPointsCordinates, double* dev_pointsVelocityArr, double* dev_initPointsCordinates);
boolean calculatePointsCoordinatesWithCuda(double time, double* nitPointsCordinates, double* pointsVelocityArr, double* pointsCordniates, int size);
cudaError_t computeWithCuda(double time, double* initPointsCordinates, double* pointsVelocityArr, double* currentPointsCordniates, int size);
#pragma once


