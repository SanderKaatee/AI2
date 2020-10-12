import random
import math

class Cluster:
    """This class represents the clusters, it contains the
    prototype and a set with the ID's (which are Integer objects)
    of the datapoints that are member of that cluster."""
    def __init__(self, dim):
        self.prototype = [0.0 for _ in range(dim)]
        self.current_members = set()

class Kohonen:
    def __init__(self, n, epochs, traindata, testdata, dim):
        self.n = n
        self.epochs = epochs
        self.traindata = traindata
        self.testdata = testdata
        self.dim = dim

        # A 2-dimensional list of clusters. Size == N x N
        self.clusters = [[Cluster(dim) for _ in range(n)] for _ in range(n)]
        # Threshold above which the corresponding html is prefetched
        self.prefetch_threshold = 0.5
        self.initial_learning_rate = 0.8
        # The accuracy and hitrate are the performance metrics (i.e. the results)
        self.accuracy = 0
        self.hitrate = 0

    def euclidean_distance(self, X,P):
        ## Euclidian distance function, formula (1) in assignment
        euc_dist = 0
        for id in range(self.dim):
                euc_dist = euc_dist + (X[id]-P[id])*(X[id]-P[id])
        return math.sqrt(euc_dist)

    def initialize_clusters(self):
        for i in range(self.n):
            for j in range(self.n):
                self.clusters[i][j].prototype = [random.uniform(0, 1) for _ in range(self.dim)]

        pass
    def calculate_square_size(self, epoch):
        size = self.n * (self.epochs - epoch) / self.epochs
        size = math.ceil(size)
        return size

    def calculate_learning_rate(self, epoch):
        size = self.initial_learning_rate * (self.epochs - epoch) / self.epochs
        return size

    def calculate_best_matching_unit(self, client):
        minimum = 200
        closest_cluster_index = None
        for i in range(self.n):
            for j in range(self.n):
                euc_dist = self.euclidean_distance(client,self.clusters[i][j].prototype)
                if (euc_dist<minimum):
                    ## Find closest cluster
                    minimum=euc_dist
                    closest_cluster_index= (i, j)            
        return closest_cluster_index

    def change_neighbourhood_nodes(self, bmu_index, client, sqr_size, learning_rate):
        bmu_neugborhood_i_start = bmu_index[0] - math.floor(sqr_size/2)
        bmu_neugborhood_i_end = bmu_index[0] + math.floor(sqr_size/2)
        bmu_neugborhood_j_start = bmu_index[1] - math.floor(sqr_size/2)
        bmu_neugborhood_j_end = bmu_index[1] + math.floor(sqr_size/2)

        for i in range(bmu_neugborhood_i_start, bmu_neugborhood_i_end):
            for j in range(bmu_neugborhood_j_start,bmu_neugborhood_j_end):
                print(i)
                print(j)
                try:
                    if i == bmu_index[0] and j == bmu_index[1]:
                        print('BMU')
                        print(bmu_index)
                    if i<0 or j<0:
                        print("Outside matrix")
                except:
                    print("Outside matrix")
                    continue
        

    def print_progress_bar(self, epoch):
        pass

    def train(self):
        # Step 1: initialize map with random vectors (A good place to do this, is in the initialisation of the clusters)
        self.initialize_clusters()
        # Repeat 'epochs' times:
        for epoch in range(self.epochs):
        #     Step 2: Calculate the square size and the learning rate, these decrease linearly with the number of epochs.
            sqr_size = self.calculate_square_size(epoch)
            learning_rate = self.calculate_learning_rate(epoch)
        #     Step 3: Every input vector is presented to the map (always in the same order)
        #     For each vector its Best Matching Unit is found, and :
            for client in self.traindata:
                bmu_index = self.calculate_best_matching_unit(client)
        #         Step 4: All nodes within the neighbourhood of the BMU are changed, you don't have to use distance relative learning.
                self.change_neighbourhood_nodes(bmu_index, client, sqr_size, learning_rate)
        # Since training kohonen maps can take quite a while, presenting the user with a progress bar would be nice

        self.print_progress_bar(epoch)
        print(epoch)
        pass

    def test(self):
        # iterate along all clients
        # for each client find the cluster of which it is a member
        # get the actual test data (the vector) of this client
        # iterate along all dimensions
        # and count prefetched htmls
        # count number of hits
        # count number of requests
        # set the global variables hitrate and accuracy to their appropriate value
        pass

    def print_test(self):
        print("Prefetch threshold =", self.prefetch_threshold)
        print("Hitrate:", self.hitrate)
        print("Accuracy:", self.accuracy)
        print("Hitrate+Accuracy =", self.hitrate+self.accuracy)
        print()

    def print_members(self):
        for i in range(self.n):
            for j in range(self.n):
                print("Members cluster["+str(i)+"]["+str(j)+"] :", self.clusters[i][j].current_members)
                print()

    def print_prototypes(self):
        for i in range(self.n):
            for j in range(self.n):
               print("Prototype cluster["+str(i)+"]["+str(j)+"] :", self.clusters[i][j].prototype)
               print()
