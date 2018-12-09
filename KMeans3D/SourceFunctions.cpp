
#include "Header.h"
#include <mpi.h>
#include <time.h>

/* Creating an MPI_Type point and cluster */
void init_new_MPI_Types(MPI_Datatype* newMPIType_Point, MPI_Datatype* newMPIType_Cluster)
{
	Point count_point;
	Cluster count_cluster;

	int array_of_blocklengths_for_point[POINT_STRUCT_SIZE] = { 1, 1, 1, 1, 1, 1, 1, 1 };
	int array_of_blocklengths_for_cluster[CLUSTER_STRUCT_SIZE] = { 1, 1, 1, 1 };

	MPI_Aint array_of_displacements_for_point[POINT_STRUCT_SIZE];
	MPI_Aint array_of_displacements_for_cluster[CLUSTER_STRUCT_SIZE];

	MPI_Datatype array_of_types_for_point[POINT_STRUCT_SIZE] = { MPI_DOUBLE, MPI_DOUBLE ,MPI_DOUBLE ,MPI_DOUBLE, MPI_DOUBLE ,MPI_DOUBLE ,MPI_INT ,MPI_INT };
	MPI_Datatype array_of_types_for_cluster[CLUSTER_STRUCT_SIZE] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE , MPI_DOUBLE };

	array_of_displacements_for_point[0] = (char *)&count_point.x - (char *)&count_point;
	array_of_displacements_for_point[1] = (char *)&count_point.y - (char *)&count_point;
	array_of_displacements_for_point[2] = (char *)&count_point.z - (char *)&count_point;
	array_of_displacements_for_point[3] = (char *)&count_point.vx - (char *)&count_point;
	array_of_displacements_for_point[4] = (char *)&count_point.vy - (char *)&count_point;
	array_of_displacements_for_point[5] = (char *)&count_point.vz - (char *)&count_point;
	array_of_displacements_for_point[6] = (char *)&count_point.currentClusterIndex - (char *)&count_point;
	array_of_displacements_for_point[7] = (char *)&count_point.previousClusterIndex - (char *)&count_point;

	array_of_displacements_for_cluster[0] = (char *)&count_cluster.x - (char *)&count_cluster;
	array_of_displacements_for_cluster[1] = (char *)&count_cluster.y - (char *)&count_cluster;
	array_of_displacements_for_cluster[2] = (char *)&count_cluster.z - (char *)&count_cluster;
	array_of_displacements_for_cluster[3] = (char *)&count_cluster.diameter - (char *)&count_cluster;

	MPI_Type_create_struct(POINT_STRUCT_SIZE, array_of_blocklengths_for_point, array_of_displacements_for_point, array_of_types_for_point, newMPIType_Point);
	MPI_Type_commit(newMPIType_Point);

	MPI_Type_create_struct(CLUSTER_STRUCT_SIZE, array_of_blocklengths_for_cluster, array_of_displacements_for_cluster, array_of_types_for_cluster, newMPIType_Cluster);
	MPI_Type_commit(newMPIType_Cluster);

}

void writeToFile(double t, double q, Cluster* clusters, int K)
{
	const char* fileName = OUTPUT_PATH;
	FILE* file;
	fopen_s(&file, fileName, "w");

	if (file == NULL) {
		printf("Couldnt open the file\n");
		exit(1);
	}
	printf("\nFirst occurrence at t = %lf with q = %lf\nCenters of the clusters:\n", t, q);
	fprintf_s(file, "First occurrence at t = %lf with q = %lf\nCenters of the clusters:\n", t, q);
	for (int i = 0; i < K; i++)
	{
		printf("%d: ( %lf , %lf , %lf )\n", i, clusters[i].x, clusters[i].y, clusters[i].z);
		fprintf_s(file, "%d: ( %lf , %lf , %lf )\n", i, clusters[i].x, clusters[i].y, clusters[i].z);
	}
	fclose(file);

}

Point* readDataFromFile(int* N, int* K, int* T, double* dt, double* LIMIT, double* QM)
{
	int i;
	const char* POINTS_FILE = INPUT_PATH;
	FILE* file = fopen(POINTS_FILE, "r");

	if (!file)
	{
		printf("could not open the file "); fflush(stdout);
		MPI_Finalize();
		exit(1);
	}

	//Getting the supplied data from file
	fscanf(file, "%d %d %d %lf %lf %lf\n", N, K, T, dt, LIMIT, QM);

	Point* points = (Point*)malloc(*N * sizeof(Point));
	checkAllocation(points);

	//Initalize points 
	for (i = 0; i < *N; i++)
	{
		fscanf(file, "%lf %lf %lf %lf %lf %lf\n", &(points[i].x), &(points[i].y), &(points[i].z), &(points[i].vx), &(points[i].vy), &(points[i].vz)); /// added z
		points[i].currentClusterIndex = 0;
		points[i].previousClusterIndex = -1;
	}

	fclose(file);
	return points;
}

/* Summing the coordinates of each point in a certain cluster */
void summarizeClusterCoordinates(Cluster* cluster, Point* points_of_current_cluster, int num_points_in_cluster, double* sumX, double* sumY, double* sumZ)//Step 3 in K-Means algorithm
{
	int i;
	*sumX = 0;
	*sumY = 0;
	*sumZ = 0;

	for (int i = 0; i < num_points_in_cluster; i++)
	{
		*sumX += points_of_current_cluster[i].x;
		*sumY += points_of_current_cluster[i].y;
		*sumZ += points_of_current_cluster[i].z;
	}
}



void checkAllocation(void* pointer)
{
	if (!pointer)
	{
		MPI_Finalize();
		exit(1);
	}
}

/*[step 1 in Boris K-Means algorithem] : assign first K points as initial clusters centers */
Cluster* getInitClusters(const Point* points, int K)
{
	int i;
	Cluster* clusters = (Cluster*)malloc(K * sizeof(Cluster));
	checkAllocation(clusters);

#pragma omp parallel for shared(clusters) 
	for (i = 0; i < K; i++)
	{
		clusters[i].x = points[i].x;
		clusters[i].y = points[i].y;
		clusters[i].z = points[i].z;
		clusters[i].diameter = 0;
	}
	return clusters;
}

int getClosestClusterIndex(double x, double y, double z, Cluster* clusters, int K)
{
	int i, closest_cluster_index = 0;
	double min_distance, current_distance;

	min_distance = calcNewDistance(x, y, z, clusters[0].x, clusters[0].y, clusters[0].z);

	for (i = 1; i < K; i++)
	{
		current_distance = calcNewDistance(x, y, z, clusters[i].x, clusters[i].y, clusters[i].z);
		if (current_distance < min_distance)
		{
			min_distance = current_distance;
			closest_cluster_index = i;
		}
	}
	return closest_cluster_index;
}

/* [step 2 in Boris K-Means Algorithm] : Group points around the cluster centers */
void assignPointsToClusters(Point** pcMatrix, int* num_points_in_cluster, Point* points, int numPointsPerProcess, Cluster* clusters, int K)
{
	int i, tid;

	/* Reset amount of points in each Cluster to zero */
#pragma omp parallel for shared(numPointsPerProcess)
	for (i = 0; i < K; i++)
	{
		num_points_in_cluster[i] = 0;
	}

	/* For each point: find a cluster center that is most close to that point  */
#pragma omp parallel for shared(points,num_points_in_cluster, pcMatrix) private(tid)
	for (i = 0; i < numPointsPerProcess; i++)
	{
		points[i].previousClusterIndex = points[i].currentClusterIndex;
		points[i].currentClusterIndex = getClosestClusterIndex(points[i].x, points[i].y, points[i].z, clusters, K);
	}

	/* Points may have been reassigned to new clusters => Update data in matrix and array of cluster sizes */
	for (i = 0; i < numPointsPerProcess; i++)
	{
		num_points_in_cluster[points[i].currentClusterIndex]++;
		pcMatrix[points[i].currentClusterIndex] = (Point*)realloc(pcMatrix[points[i].currentClusterIndex], num_points_in_cluster[points[i].currentClusterIndex] * sizeof(Point));
		pcMatrix[points[i].currentClusterIndex][(num_points_in_cluster[points[i].currentClusterIndex]) - 1] = points[i];
	}
}

double calcNewDistance(double x1, double y1, double z1, double x2, double y2, double z2)
{
	return sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2) + pow((z1 - z2), 2));
}

/* [step 3 in Boris K-Means Algorithm] : Recalculate the cluster centers */
void calcNewClusterCenter(Cluster* cluster, Point* points_of_current_cluster, int num_points_in_cluster)
{
	int i;
	double sumX = 0, sumY = 0, sumZ = 0;

	/* The new X coordinate of a cluster center is the average of all X coordinates of all the points in that cluster => Same for Y and Z*/
	for (int i = 0; i < num_points_in_cluster; i++)
	{
		sumX += points_of_current_cluster[i].x;
		sumY += points_of_current_cluster[i].y;
		sumZ += points_of_current_cluster[i].z;
	}

	cluster->x = (sumX / num_points_in_cluster);
	cluster->y = (sumY / num_points_in_cluster);
	cluster->z = (sumZ / num_points_in_cluster);
}

/* Cluster diametr is the maximum distance between 2 points in given cluster to find
maximum distance we check the distance from each point in that cluster to every other point */
double calcNewClusterDiameter(Point* points_of_current_cluster, int num_points_in_cluster)
{

	int i, j;
	double max_distance = 0, current_distance = 0;

	for (i = 0; i < num_points_in_cluster; i++)
	{
		for (j = 1; j < num_points_in_cluster; j++)
		{
			current_distance = calcNewDistance(points_of_current_cluster[i].x, points_of_current_cluster[i].y, points_of_current_cluster[i].z, points_of_current_cluster[j].x, points_of_current_cluster[j].y, points_of_current_cluster[j].z);

			if (max_distance < current_distance)
			{
				max_distance = current_distance;
			}
		}
	}
	return max_distance;
}

/* [step 6 in Boris K-Means Algorithm] : Evaluate the quality of the clusters found */
double getEvaluatedQuality(Point** pcMatrix, Cluster* clusters, int K, int* num_points_in_cluster)
{
	/*
	The Quality is equal to an average of ratio diameters of the cluster divided by distance to other clusters.
    For K = 3 the quality is: q = (d1/D12 + d1/D13 + d2/D21 + d2/D23 + d3/D31 + d3/D32) / 6
    where di is a diameter of cluster i and Dij is a distance between centers of cluster i and cluster j 
	*/
	double quality = 0, Dij = 0, SigmaRatio = 0, num_of_sigma_elements;
	int i, j;

#pragma omp parallel for shared(clusters) private(j)
	for (i = 0; i < K; i++)
	{
		/* calculate di */
		clusters[i].diameter = calcNewClusterDiameter(pcMatrix[i], num_points_in_cluster[i]);

#pragma omp parallel for
		for (j = 0; j < K; j++)
		{
			if (i != j)
			{
				/* calculate Dij */
				Dij = calcNewDistance(clusters[i].x, clusters[i].y, clusters[i].z, clusters[j].x, clusters[j].y, clusters[j].z);
				/* add new ratio to the summation expression (Sigma) */
				SigmaRatio += clusters[i].diameter / Dij;
			}
		}
	}

	num_of_sigma_elements = K * (K - 1);
	quality = SigmaRatio / num_of_sigma_elements;
	return quality;
}

double runKMeansAlgorithm(Point* points, Cluster* clusters, Point** pcMatrix, int* num_points_in_cluster, int numPointsPerProcess, int K,
	int LIMIT, double QM, double T, double dt, MPI_Datatype MPI_Point, MPI_Datatype MPI_Cluster, double* array_of_coordinates_summations,
	int numprocs, double* time, double* array_of_current_points_coordinates, double* array_of_initial_points_coordinates, double* array_of_points_velocities)
{
	double n, current_quality, quality = 0;
	int rank;

	for (*time = 0, n = 0; n < T / dt; n++)
	{
		*time = n*dt;
	
		// MASTER calculated 'time' and sent to all Slaves 
#pragma omp parallel for 
		for (rank = 1; rank < numprocs; rank++)
			MPI_Send(time, 1, MPI_DOUBLE, rank, TRANSFER_TAG, MPI_COMM_WORLD);

		// Calculate new points coordinated using CUDA 
		calculatePointsCoordinatesWithCuda(*time, array_of_initial_points_coordinates, array_of_current_points_coordinates, array_of_points_velocities, numPointsPerProcess * DIM);
		updateCoordinatesInPointsArray(points, numPointsPerProcess, array_of_current_points_coordinates);

		//Calculate new points coordinated without using CUDA	
		//calcPointsCoordinatesWithoutCuda(points, numPointsPerProcess, *time);

		//Master runs K-Mean Algorithm
		current_quality = kMeansAlgorithmMaster(points, clusters, pcMatrix, num_points_in_cluster, numPointsPerProcess, K,
			LIMIT, numprocs, MPI_Point, MPI_Cluster, array_of_coordinates_summations);

		//if quality < QM we stop
		if (current_quality < QM)
		{
			//Master alerting all Slaves to terminate
#pragma omp parallel for  
			for (rank = 1; rank < numprocs; rank++)
				MPI_Send(time, 1, MPI_DOUBLE, rank, FINAL_TERMINATION_TAG, MPI_COMM_WORLD);

			return current_quality;
		}

		//checks if the current quality is better than the best attained quality so far.
		if (current_quality < quality || quality == 0)
			quality = current_quality;
	}

	//Master alerting all Slaves to terminate (after we finished iterating on Time Interval )
#pragma omp parallel for 
	for (rank = 1; rank < numprocs; rank++)
		MPI_Send(time, 1, MPI_DOUBLE, rank, FINAL_TERMINATION_TAG, MPI_COMM_WORLD);

	return quality;
}

double kMeansAlgorithmMaster(Point* points, Cluster* clusters, Point** pcMatrix, int* num_points_in_cluster, int numPointsPerProcess, int K,
	int LIMIT, int numprocs, MPI_Datatype MPI_Point, MPI_Datatype MPI_Cluster, double* array_of_coordinates_summations)
{

	MPI_Status status;
	int i, j, z, k, flag, current_size = 0, previous_size, flagCounter = 0;
	double* array_of_all_sums;
	int* array_of_total_clusters_sizes;

	array_of_all_sums = (double*)calloc(K * DIM, sizeof(double));
	array_of_total_clusters_sizes = (int*)calloc(K, sizeof(int));

	//for each point, reset index of it's previos cluster  
#pragma omp parallel for shared(points)
	for (i = 0; i < numPointsPerProcess; i++)
		points[i].previousClusterIndex = -1;

	
	for (i = 0; i < LIMIT; i++)
	{
		
		flag = 0;
		flagCounter = 0;

		//Send Clusters to Slaves
#pragma omp parallel for 
		for (int rank = 1; rank < numprocs; rank++)
			MPI_Send(clusters, K, MPI_Cluster, rank, TRANSFER_TAG, MPI_COMM_WORLD);

		// [step 2 in Boris K-Means Algorithm] : Group points around the given cluster centers
		assignPointsToClusters(pcMatrix, num_points_in_cluster, points, numPointsPerProcess, clusters, K);
		
		// [step 3 in Boris K-Means Algorithm] : Recalculate the cluster centers 
#pragma omp parallel for shared(clusters,pcMatrix,num_points_in_cluster,array_of_coordinates_summations)
		for (j = 0; j < K; j++)
		{
			//summing all points coordinates in the clusters that Master process posess
			summarizeClusterCoordinates(clusters + j, pcMatrix[j], num_points_in_cluster[j], array_of_coordinates_summations + (j * DIM),
				array_of_coordinates_summations + ((j * DIM) + 1), array_of_coordinates_summations + ((j * DIM) + 2));

		}
		
		//getting summation of points in Slaves clusters
		MPI_Reduce(array_of_coordinates_summations, array_of_all_sums, K * DIM, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
		MPI_Reduce(num_points_in_cluster, array_of_total_clusters_sizes, K, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
		
		//new cluster centers are average of all points in the cluster
#pragma omp parallel for shared(clusters)
		for (j = 0; j < K; j++)
		{
			clusters[j].x = array_of_all_sums[(j * DIM)] / array_of_total_clusters_sizes[j];
			clusters[j].y = array_of_all_sums[(j * DIM) + 1] / array_of_total_clusters_sizes[j];
			clusters[j].z = array_of_all_sums[(j * DIM) + 2] / array_of_total_clusters_sizes[j];
		}
		
		//[step 4 in Boris K-Means Algorithm] : Check the termination condition – no points move to other clusters
		for (j = 0; j < numPointsPerProcess && (points[j].currentClusterIndex == points[j].previousClusterIndex); j++);
		flag = (j == numPointsPerProcess ? 1 : 0);
		
		//Send the answer to termination condition query 
		MPI_Reduce(&flag, &flagCounter, 1, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
		
		//[step 5 in Boris K-Means Algorithm] : Check if the termination condition fulfills
		//If all slaves reached terminetion condition => they raised a flag =>  flagCounter's value will be the same as numprocs
		if (flagCounter == numprocs)
		{
			
#pragma omp parallel for 
			for (int procsNumber = 1; procsNumber < numprocs; procsNumber++)
				MPI_Send(clusters, K, MPI_Cluster, procsNumber, MID_TERMINATION_TAG, MPI_COMM_WORLD);

			//Gather all the points from all the process to the Master process
			returnAllPointsToMaster(pcMatrix, num_points_in_cluster, array_of_total_clusters_sizes, K, numprocs, MPI_Point);

			free(array_of_total_clusters_sizes);
			free(array_of_all_sums);

			return getEvaluatedQuality(pcMatrix, clusters, K, num_points_in_cluster);
		}

	}
	for (int procsNumber = 1; procsNumber < numprocs; procsNumber++)
		MPI_Send(clusters, K, MPI_Cluster, procsNumber, MID_TERMINATION_TAG, MPI_COMM_WORLD);

	//Gather all the points from all slave processes to the Master process
	returnAllPointsToMaster(pcMatrix, num_points_in_cluster, array_of_total_clusters_sizes, K, numprocs, MPI_Point);

	free(array_of_total_clusters_sizes);
	free(array_of_all_sums);
	return getEvaluatedQuality(pcMatrix, clusters, K, num_points_in_cluster);

}

/* Slaves perform K-Means Algorithm */
void kMeansAlgorithmSlave(Point* points, Cluster* clusters, Point** pcMatrix, int* num_points_in_cluster, int numPointsPerProcess, int K,
	MPI_Datatype MPI_Point, MPI_Datatype MPI_Cluster, double* array_of_coordinates_summations)
{
	MPI_Status status;
	status.MPI_TAG = TRANSFER_TAG;
	int j, flag;

	while (status.MPI_TAG == TRANSFER_TAG)
	{
		flag = 0;

		//Each Slave get clusters from Master
		MPI_Recv(clusters, K, MPI_Cluster, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		if (status.MPI_TAG == MID_TERMINATION_TAG)
		{
			getPointsFromMaster(K, num_points_in_cluster, pcMatrix, MPI_Point);
		}
		else
		{
			// [step 2 in Boris K-Means Algorithm] : Group points around the given cluster centers
			assignPointsToClusters(pcMatrix, num_points_in_cluster, points, numPointsPerProcess, clusters, K);

			// [step 3 in Boris K-Means Algorithm] : Recalculate the cluster centers 
#pragma omp parallel for shared(clusters,pcMatrix,num_points_in_cluster,array_of_coordinates_summations)
			for (j = 0; j < K; j++)
			{
				summarizeClusterCoordinates(clusters + j, pcMatrix[j], num_points_in_cluster[j], array_of_coordinates_summations + (j * DIM),
					array_of_coordinates_summations + ((j * DIM) + 1), array_of_coordinates_summations + ((j * DIM) + 2));
			}

			//Each Slave returns their coordinates summations to Master
			MPI_Reduce(array_of_coordinates_summations, array_of_coordinates_summations, K * DIM, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
			MPI_Reduce(num_points_in_cluster, num_points_in_cluster, K, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);

			// [step 4 in Boris K-Means Algorithm] : check if no points have moved to other clusters 
			for (j = 0; j < numPointsPerProcess && (points[j].currentClusterIndex == points[j].previousClusterIndex); j++);
			flag = (j == numPointsPerProcess ? 1 : 0);

			//Each Slave sends the value of their flag to Master
			MPI_Reduce(&flag, &flag, 1, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
		}

	}
}

/* Each slave sends it's points to Master*/
void returnAllPointsToMaster(Point** pcMatrix, int* clustersSize, int* totalClustersSize, int K, int numprocs, MPI_Datatype MPI_Point)
{
	MPI_Status status;
	int i, j, previous_size, current_size;

	for (i = 0; i < K; i++)
	{
		pcMatrix[i] = (Point*)realloc(pcMatrix[i], totalClustersSize[i] * sizeof(Point));

		for (j = 1; j < numprocs; j++)
		{
			previous_size = clustersSize[i];
			MPI_Recv(&current_size, 1, MPI_INT, j, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			clustersSize[i] += current_size;
			MPI_Recv(pcMatrix[i] + previous_size, current_size, MPI_Point, j, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		}
	}
}

/* Function for the slaves to recieve their points from Master process*/
void getPointsFromMaster(int K, int* num_points_in_cluster, Point** pcMatrix, MPI_Datatype MPI_Point)
{
	for (int i = 0; i < K; i++)
	{
		MPI_Send(&(num_points_in_cluster[i]), 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
		MPI_Send(pcMatrix[i], num_points_in_cluster[i], MPI_Point, MASTER, 0, MPI_COMM_WORLD);
	}
}

/* Slaves start participating in K-Means*/
void slaveDoWork(Point* points, Cluster* clusters, Point** pcMatrix, int* num_points_in_cluster, int numPointsPerProcess, int K,
	MPI_Datatype MPI_Point, MPI_Datatype MPI_Cluster, double* array_of_coordinates_summations, double* array_of_current_points_coordinates, double * array_of_initial_points_coordinates, double* array_of_points_velocities)
{

	MPI_Status status;
	status.MPI_TAG = TRANSFER_TAG;
	double time;
	int i;

	while (status.MPI_TAG != FINAL_TERMINATION_TAG)
	{
		MPI_Recv(&time, 1, MPI_DOUBLE, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		if (status.MPI_TAG != FINAL_TERMINATION_TAG)
		{
			//calculate new points coordinates using CUDA
			calculatePointsCoordinatesWithCuda(time, array_of_initial_points_coordinates, array_of_points_velocities, array_of_current_points_coordinates, numPointsPerProcess * DIM);
			updateCoordinatesInPointsArray(points, numPointsPerProcess, array_of_current_points_coordinates);

			//without using CUDA	
			//calcPointsCoordinatesWithoutCuda(points, numPointsPerProcess, time);

#pragma omp parallel for shared(points)
			for (i = 0; i < numPointsPerProcess; i++)
				points[i].previousClusterIndex = -1;

			kMeansAlgorithmSlave(points, clusters, pcMatrix, num_points_in_cluster, numPointsPerProcess, K, MPI_Point, MPI_Cluster, array_of_coordinates_summations);
		}
		else
		{
			break;
		}
	}
}

void initCoordVelocArrays(double* array_of_current_points_coordinates, double* array_of_initial_points_coordinates, double* array_of_points_velocities, Point* points, int numPointsPerProcess)
{
#pragma omp parallel for shared(array_of_current_points_coordinates,array_of_initial_points_coordinates,array_of_points_velocities)
	for (int i = 0; i < numPointsPerProcess; i++)
	{
		/* because points havent moved yet, we assign current and initial points coordinates to be the same*/
		array_of_initial_points_coordinates[i * DIM] = points[i].x;
		array_of_initial_points_coordinates[(i * DIM) + 1] = points[i].y;
		array_of_initial_points_coordinates[(i * DIM) + 2] = points[i].z;

		array_of_current_points_coordinates[i * DIM] = points[i].x;
		array_of_current_points_coordinates[(i * DIM) + 1] = points[i].y;
		array_of_current_points_coordinates[(i * DIM) + 2] = points[i].z;

		array_of_points_velocities[i * DIM] = points[i].vx;
		array_of_points_velocities[(i * DIM) + 1] = points[i].vy;
		array_of_points_velocities[(i * DIM) + 2] = points[i].vz;
	}
}

void calcPointsCoordinatesWithoutCuda(Point* points, int numPointsPerProcess, double t)
{
	int i;
#pragma omp parallel for shared(points)
	for (i = 0; i < numPointsPerProcess; i++)
	{
		points[i].x = points[i].x + (t*points[i].vx);
		points[i].y = points[i].y + (t*points[i].vy);
		points[i].z = points[i].z + (t*points[i].vz);
	}
}

void updateCoordinatesInPointsArray(Point* points, int numPointsPerProcess, double* array_of_current_points_coordinates)
{
	int i;
#pragma omp parallel for shared(points)
	for (i = 0; i <numPointsPerProcess; i++)
	{
		points[i].x = array_of_current_points_coordinates[(i * DIM)];
		points[i].y = array_of_current_points_coordinates[(i * DIM) + 1];
		points[i].z = array_of_current_points_coordinates[(i * DIM) + 2];
	}
}
