"""kmeans.py"""
import random
import math
class Cluster:
    """This class represents the clusters, it contains the
    prototype (the mean of all it's members) and memberlists with the
    ID's (which are Integer objects) of the datapoints that are member
    of that cluster. You also want to remember the previous members so
    you can check if the clusters are stable."""
    def __init__(self, dim):
        self.prototype = [0.0 for _ in range(dim)]
        self.current_members = set()
        self.previous_members = set()


class KMeans:

    def no_primitives_change(self, clusters):
        ## If there is a change in the clusters (previous != current), we return False
        for cluster in clusters:
            if cluster.previous_members != cluster.current_members:
                return False
        
        ## No change found so we return true
        return True

    def euclidean_distance(self, X,P):
        ## Euclidian distance function, formula (1) in assignment
        euc_dist = 0
        for idx in range(self.dim):
                euc_dist = euc_dist + (X[idx]-P[idx])*(X[idx]-P[idx])
        return math.sqrt(euc_dist)

    def generate_partition(self, client, clusters):
        ## Function to generate the partition, aka step 2 of the assignment
        for cluster in clusters:
            ## Empty our clusters while making sure we know what the previous
            ## members were
            cluster.previous_members = cluster.current_members
            cluster.current_members.clear()

        ## For each client we check which is the closest cluster and then add that client to that cluster
        for client_id in range(len(client)):
            ## Euclidian distance will be at most 200
            minimum = 200
            closest_cluster = None
            for cluster in clusters:
                ## The prototype is the cluster center
                euc_dist = self.euclidean_distance(client[client_id],cluster.prototype)
                if (euc_dist<minimum):
                    ## Find closest cluster
                    minimum=euc_dist
                    closest_cluster=cluster
            closest_cluster.current_members.add(client_id)

    def recalculate_cluster_centers(self,client, clusters):
        ## Function to (re)calculate the cluster center, aka step 3 of the assignment

        for cluster in clusters:
            number_of_members=len(cluster.current_members)

            new_cluster_center = [0.0 for _ in range(self.dim)]
            for client_id in cluster.current_members:
                for id in range(self.dim):
                    new_cluster_center[id] = new_cluster_center[id] + client[client_id][id]
            
            if (number_of_members != 0):
                for index in range(len(new_cluster_center)):
                    new_cluster_center[index] = new_cluster_center[index]/number_of_members
            cluster.prototype=new_cluster_center



    def __init__(self, k, traindata, testdata, dim):
        self.k = k
        self.traindata = traindata
        print(self.traindata)
        self.testdata = testdata
        self.dim = dim

        # Threshold above which the corresponding html is prefetched
        self.prefetch_threshold = 0.5
        # An initialized list of k clusters
        self.clusters = [Cluster(dim) for _ in range(k)]

        # The accuracy and hitrate are the performance metrics (i.e. the results)
        self.accuracy = 0
        self.hitrate = 0

    def train(self):
        # implement k-means algorithm here:
        # Step 1: Select an initial random partioning with k clusters
        for cluster in self.clusters:
            cluster.prototype = [random.uniform(0, 1) for _ in range(self.dim)]

        iteration = 0

        while True:
            # Step 2: Generate a new partition by assigning each datapoint to its closest cluster center
            self.generate_partition(self.traindata, self.clusters)

            # Step 3: recalculate cluster  centers
            self.recalculate_cluster_centers(self.traindata, self.clusters)

            # Step 4: repeat until clustermembership stabilizes

            print(iteration)
            if(self.no_primitives_change(self.clusters)):
                break

            ## Break the while-loop if it gets stuck
            iteration=iteration+1
            if(iteration==1000):
                print("Something went wrong: the clusters do not stabalize")
                break        
            pass

    def test(self):
        prefetched_htmls = 0
        hits = 0
        requests = 0
        # for each client find the cluster of which it is a member
        for client_id in range(len(self.traindata)):
            for cluster in self.clusters:
                if client_id in cluster.current_members:
                    # get the actual testData (the vector) of this client
                    testdata = self.testdata[client_id]
                    # iterate along all dimensions
                    for idx in range(self.dim):
                        prefetch = False
                        request = False
                        # and count prefetched htmls
                        if testdata[idx] == 1:
                            requests = requests + 1
                            request = True
                        
                        if cluster.prototype[idx] > self.prefetch_threshold:
                            prefetched_htmls = prefetched_htmls + 1
                            prefetch = True

                        if prefetch == True and request == True:
                            hits = hits + 1
        
        hitrate = hits / requests
        accuracy = hits / prefetched_htmls
        
        print("hitrate:")
        print(round(hitrate, 2))
        print("accuracy:")
        print(round(accuracy, 2))













       
        # count number of hits
        # count number of requests
        # set the variables hitrate and accuracy to their appropriate value
        pass


    def print_test(self):
        print("Prefetch threshold =", self.prefetch_threshold)
        print("Hitrate:", self.hitrate)
        print("Accuracy:", self.accuracy)
        print("Hitrate+Accuracy =", self.hitrate+self.accuracy)
        print()

    def print_members(self):
        for i, cluster in enumerate(self.clusters):
            print("Members cluster["+str(i)+"] :", cluster.current_members)
            print()

    def print_prototypes(self):
        for i, cluster in enumerate(self.clusters):
            print("Prototype cluster["+str(i)+"] :", cluster.prototype)
            print()
