Project by: Victoria Glazunov


Parallel implementation of K-Means

  
Problem Definition

Given a set of points in 3-dimensional space. Initial position (xi, yi, zi) and velocity (vxi, vyi, vzi) are known for each point Pi. Its position at the given time t can be calculated as follows:
xi(t) = xi + t*vxi
yi(t) = yi + t*vyi
zi(t) = zi + t*vzi

I implemented a simplified K-Means algorithm to find K clusters. The program findes a first occurrence during given time interval [0, T] when a system of K clusters has a Quality Measure q that is less than given value QM.


Additional info:

•	Proper parallel implementation of the problem with three components: MPI+OpenMP+CUDA configuration
•	Implementation of the K-MEANS algorithm explained below.  
•	Used first K points at t=0 as initial positions of the centers of the clusters. 
	In case that in some iteration there will be no points in cluster – kept its center for the next iteration.
•	Calculations stops after the Quality Measure was reached – is less than given value QM
•	Diameter of a cluster is the largest distance between two points of this cluster.
•	Distance between two clusters is a distance between centers of these clusters. 
•	Project is able to run on more that one computer (using MPICH )
•	The set of points may contain at least 10000 but not more than 3000000 points.  


Simplified K-Means algorithm

1.	Choose first K points as centers of clusters.
2.	Group points around the given cluster centers - for each point define a center that is most close to the point.
3.	Recalculate the cluster centers (average of all points in the cluster)
4.	Check the termination condition – no points move to other clusters or maximum iteration LIMIT was made.
5.	Repeat from 2 till the termination condition fulfills.
6.	Evaluate the Quality of the clusters found. Calculate diameter of each cluster – maximum distance between any two points in this 	 cluster. The Quality is equal to an average of ratio diameters of the cluster divided by distance to other clusters. For 		example, in case of k = 3 the quality is equal 
	q = (d1/D12 + d1/D13 + d2/D21 + d2/D23 + d3/D31 + d3/D32) / 6, 
	where di is a diameter of cluster i and Dij is a distance between centers of cluster i and cluster j.




Input data and Output Result of the project
Supplied input data 
•	N - number of points
•	K - number of clusters to find
•	LIMIT – the maximum number of iterations for K-MEAN algorithm. 
•	QM – quality measure to stop
•	T – defines the end of time interval [0, T]
•	dT – defines moments t = n*dT, n = { 0, 1, 2, … , T/dT} for which calculate the clusters and the quality
•	Coordinates and Velocities of all points

Input File format

The first line of the file contains   N    K    T   dT   LIMIT   QM.  
Next lines are Initial Positions and Velocities of the points (xi, yi, zi, vxi, vyi, vzi)

For example:
50000    42    30    0.1    200.0       7.3
2.3      4. 5      6. 55     -2.3   13.3   1.1
76.2   -3.56    50.0        12    -0.7    22.3
 …
45.23   20      -167.1    98.0  99.2  -113.2 



Output File format

The output file contains information on the found clusters with the moment when the Quality Measure QM is reached for first time.

Like that:
First occurrence t = value  with q = value
Centers of the clusters:
x1   y1   z1
x2   y2   z2

x3   y3   z3

x4   y4   z4

For example, in case K = 4:
First occurrence t = 24.5  with q = 6.9
Centers of the clusters:
1.123     34   13.2
-5.3       17.01    90.4
33.56    -23   -1.3
14.1      98    14.9






