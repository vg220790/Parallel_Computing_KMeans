
#include "Header.h"
#include <mpi.h>
#include <omp.h>

int main(int argc, char *argv[])
{
	/*
	N		- total number of points  (constant)
	K		- number of clusters	  (constant)
	LIMIT	- max iterations		  (constant)
	QM		- quality max limit	      (constant)
	T		- end of time interval    (constant)
	dt		- time at current moment  (variable)
	quality - quality at curr moment  (variable)
	*/
	int N, K, T, numPointsPerProcess;
	double  dt, LIMIT, QM, quality;


	/*
	points	    - array of points
	clusters	- array of clusters
	pcMatrix	- Points by Clusters Matrix
				  Index of the row is the cluster number
				  Elements in row are the points of that cluster
	
	num_points_in_cluster - array of clusters sizes
	each element contains size of cluster with same indaex
	*/
	Point* points;
	Cluster* clusters;
	Point** pcMatrix;
	int* num_points_in_cluster;
	double* array_of_initial_points_coordinates, *array_of_current_points_coordinates,  *array_of_points_velocities, *array_of_coordinates_summations;

	int  namelen, numprocs, myId, i;
	char processor_name[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myId);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Get_processor_name(processor_name, &namelen);
	MPI_Status status;

	MPI_Datatype MPI_Point;
	MPI_Datatype MPI_Cluster;
	init_new_MPI_Types(&MPI_Point, &MPI_Cluster);

	/* Master */
	if (myId == MASTER)
	{
		/* read data from file */
		points = readDataFromFile(&N, &K, &T, &dt, &LIMIT, &QM);  // changed the function a bit
		printf("\n N = %d, K = %d, T = %d, dt = %lf, LIMIT = %lf, QM = %lf\n", N, K, T, dt, LIMIT, QM);
		/* assign an even amount of points to each process*/
		numPointsPerProcess = N / numprocs;

		/*[step 1 in Boris K-Means algorithem] : assign first K points as initial clusters centers */
		clusters = getInitClusters(points, K);

		/* Master broadcasting values of K and numPointsPerProcess to each Slave*/
		MPI_Bcast(&K, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
		MPI_Bcast(&numPointsPerProcess, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

		/* If after splitting the points between processes, there is an unhandled remainder of points
		then those points will be assigned to Master process */
		numPointsPerProcess = (N / numprocs) + (N % numprocs);;
	}

	/* Slaves */
	else
	{
		/* each Slave recieves from Master value of 'K' and amount of points they were assigned */
		MPI_Bcast(&K, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
		MPI_Bcast(&numPointsPerProcess, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	}

	/* Allocating memory for arrays and matrix */
	pcMatrix = (Point**)calloc(K, sizeof(Point*));
	checkAllocation(pcMatrix);
	num_points_in_cluster = (int*)calloc(K, sizeof(int));
	checkAllocation(num_points_in_cluster);
	array_of_current_points_coordinates = (double*)calloc(numPointsPerProcess * DIM, sizeof(double));
	checkAllocation(array_of_current_points_coordinates);
	array_of_initial_points_coordinates = (double*)calloc(numPointsPerProcess * DIM, sizeof(double));
	checkAllocation(array_of_initial_points_coordinates);
	array_of_points_velocities = (double*)calloc(numPointsPerProcess * DIM, sizeof(double));
	checkAllocation(array_of_points_velocities);
	array_of_coordinates_summations = (double*)calloc(K * DIM, sizeof(double));
	checkAllocation(array_of_coordinates_summations);

	if (myId == MASTER)
	{
		double start_time, end_time, time;
		int numOfPoints_Master = numPointsPerProcess;
		int numOfPoints_Slave = numPointsPerProcess - (N % numprocs);

		for (int rank = 1; rank < numprocs; rank++)
		{
			/* Master sends to each Slave their block of points from 'points' array */
			MPI_Send(points + numOfPoints_Master + (numOfPoints_Slave * (rank - 1)),
				numOfPoints_Slave, MPI_Point, rank, 0, MPI_COMM_WORLD);
		}

		printf("STARTING K-MEANS\n"); fflush(stdout);
		start_time = omp_get_wtime();

		/* Master - initializing data of points coordinates and velocities in arrays*/
		initCoordVelocArrays(array_of_current_points_coordinates, array_of_initial_points_coordinates,
			array_of_points_velocities, points, numPointsPerProcess);

		/* Starting K-Means algorithm*/
		quality = runKMeansAlgorithm(points, clusters, pcMatrix, num_points_in_cluster, numPointsPerProcess, K, LIMIT, QM, T, dt, MPI_Point, MPI_Cluster, 
		array_of_coordinates_summations, numprocs, &time, array_of_current_points_coordinates, array_of_initial_points_coordinates, array_of_points_velocities);

		end_time = omp_get_wtime();

		/* writing results to file */
		writeToFile(time, quality, clusters, K);

		printf("The quality is : %lf\n", quality); fflush(stdout);
		printf("Time %g\n", end_time - start_time);
	}
	else//Slaves
	{
		points = (Point*)calloc(numPointsPerProcess, sizeof(Point));
		checkAllocation(points);
		clusters = (Cluster*)calloc(K, sizeof(Cluster));
		checkAllocation(clusters);

		/* Each Slave gets from Master their block of points from 'points' array */
		MPI_Recv(points, numPointsPerProcess, MPI_Point, MASTER, 0, MPI_COMM_WORLD, &status);

		/* Slaves - initializing data of points coordinates and velocities in arrays*/
		initCoordVelocArrays(array_of_current_points_coordinates, array_of_initial_points_coordinates, array_of_points_velocities, points, numPointsPerProcess);
		
		slaveDoWork(points, clusters, pcMatrix, num_points_in_cluster, numPointsPerProcess,
			K, MPI_Point, MPI_Cluster, array_of_coordinates_summations, array_of_current_points_coordinates,
			array_of_initial_points_coordinates, array_of_points_velocities);

	}

	/* Freeing memory at the end of the program*/
	free(num_points_in_cluster);
	for (int i = 0; i < K; i++)
	{
		free(pcMatrix[i]);
	}
	free(pcMatrix);
	free(clusters);
	free(points);
	free(array_of_coordinates_summations);
	free(array_of_current_points_coordinates);
	free(array_of_initial_points_coordinates);
	free(array_of_points_velocities);

	printf("Process %d ended\n", myId);
	fflush(stdout);
	MPI_Finalize();
	return 0;
}





