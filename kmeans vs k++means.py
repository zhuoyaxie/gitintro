"""
pset1
Implement the k-means algorithm in a language of your choice (MATLAB, Python, or R are recommended),
initializing the cluster centers randomly, as explained in the slides. The algorithm should
terminate when the cluster assignments (and hence, the centroids) don’t change anymore.

a. The ﬁle contains 1797 8 × 8 pixel grey scale images of handwritten digits equally distributed among the digits 0, 1, ..., 9. 
Each image has been ﬂattened to a 64 dimensional vector
Each row of this text ﬁle represents one image

Test your code on this data and plot the resulting cluster centers as an image - your results
should look like the digits 0, 1, ..., 9. 
Note that because of the random initialization, different runs may produce diﬀerent results, and in some cases some of cluster centers might not look like digits.
Plot the value of the distortion function as a function of iteration number over 20 separateruns(different random seeds) of the algorithm on the same plot. 
Comment on the plot. You may ﬁnd it helpful to divide the points by 255 so that each pixel entry is now between 0 and 1 before running kmeans.

"""
####################################################################
## Data Cleaning and functions
####################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def random_init(data, k):
    '''
    K random points based on initial subset
    '''
    n,p=data.shape
    points=np.random.choice(np.arange(n),k) 
    starting_points=data[np.ix_(list(points), list(np.arange(p)))] #use numpy to merge two 1d objects into NxN
    return starting_points

def dist(x,y):
    '''euclidiean distance function'''
    return np.sqrt(np.sum((x-y)**2))

####################################################################
## Part A: implenting k means
####################################################################

def kmeans(data,k,max_iteration=20):
    '''
    kmeans function which calls upon random_init (to initialize) and dist (to calculate euclidiean distance)
    ----
    step 1: randomly generate points for center
    step 2: calculate distance between all points and center
    step 3: update the centers
    step 4: repeat step 2-3 until centers do not change
    
    Return centriod dictionary and distortion
    '''
    distortion_sum=[]
    
    #step 1: starting means... 
    means=random_init(data, k)
    
    for i in range(max_iteration): #loop through iterations
        centriod_dict={tuple(mean): [] for mean in means.tolist()} #empty dictionary with means as keys to see which points are closest eucliean distance...         
        #step 2: find the distance between centers and point
        for x in data:
            distances={dist(x,center):tuple(center) for center in means.tolist()} #dictionary with distances... distance as key and center as value
            centriod_dict[distances[min(distances.keys())]].append(x)
        
        #step 3: update the centers
        old_means=means #store previous values
        means=[]
        for mean in old_means: 
            centriod_amount = len(centriod_dict[tuple(mean)]) #amount of points in cluster
            new_center = (np.array(centriod_dict[tuple(mean)]).sum(0) / centriod_amount) #mean of each value in the cluster
            means.append(new_center)
        means = np.array(means)

        #for distortions
        distortion = 0
        for key in centriod_dict:
            for value in centriod_dict[key]:
                distortion += dist(key, value)
        distortion_sum.append(distortion)

        if np.array_equal(old_means, means): #breaks the code if new centriods and previous centriods are the same
            break
        
    return centriod_dict, distortion_sum


def kmeanplusplus_init(data,k):
    '''
    Choose cluster by cumulative probability
    '''
    n,p=data.shape
    rand_points=[np.random.choice(np.arange(n),1)] #random point
    while len(rand_points)!=k:
        nums=[]
        for x in data:
            centriod_amount=[]
            for point in rand_points:
                centriod_amount.append([dist(x,data[np.ix_(point,list(np.arange(p)))][0]),point])
            smallest_dist = min(centriod_amount)
            nums.append(smallest_dist[0]**2)
        nums=np.array(nums)
        denoms=nums.sum()
        rand_points.append(np.random.choice(np.arange(n),1,p=nums/denoms))
    starting_points=data[np.ix_(np.array(rand_points).T[0],list(p))]
    return starting_points


def kmeansplusplus(data,k,max_iteration=20):
    '''
    kmeans function which calls upon random_init (to initialize) and dist (to calculate euclidiean distance)
    ----
    step 1: keamsplusplus formula for center
    step 2: calculate distance between all points and center
    step 3: update the centers
    step 4: repeat step 2-3 until centers do not change
    
    Return centriod dictionary and distortion
    '''
    distortion_sum=[]
    
    #step 1: starting means... 
    means=kmeanplusplus_init(data, k)
    
    for i in range(max_iteration): #loop through iterations
        centriod_dict={tuple(mean): [] for mean in means.tolist()} #empty dictionary with means as keys to see which points are closest eucliean distance...         
        #step 2: find the distance between centers and point
        for x in data:
            distances={dist(x,center):tuple(center) for center in means.tolist()} #dictionary with distances... distance as key and center as value
            centriod_dict[distances[min(distances.keys())]].append(x)
        
        #step 3: update the centers
        old_means=means #store previous values
        means=[]
        for mean in old_means: 
            centriod_amount = len(centriod_dict[tuple(mean)]) #amount of points in cluster
            new_center = (np.array(centriod_dict[tuple(mean)]).sum(0) / centriod_amount) #mean of each value in the cluster
            means.append(new_center)
        means = np.array(means)

        #for distortions
        distortion = 0
        for key in centriod_dict:
            for value in centriod_dict[key]:
                distortion += dist(key, value)
        distortion_sum.append(distortion)

        if np.array_equal(old_means, means): #breaks the code if new centriods and previous centriods are the same
            break
        
    return centriod_dict, distortion_sum

if __name__=='__main__':
    #### load and visualize
    mnist = np.loadtxt('mnist_small.txt')/255
    plt.imshow(mnist[0].reshape(8,8))
    
    ####kmeans
    centriod_dict, distortion_sum=kmeans(mnist,k=10,max_iteration=20)
    #plot resulting cluster centers as image... these look fine
    for key in centriod_dict.keys(): 
         plt.imshow(np.array(key).reshape(8,8))
         plt.show()
    #distortion function with 20 iterations
    distortion_frame=pd.DataFrame(index=range(20),columns=range(20))
    for i in range(20):
        print(i)
        centriod_dict, distortion_sum=kmeans(mnist,k=10,max_iteration=20)
        distortion_frame[i]=distortion_sum+[np.nan]*(20-len(distortion_sum))        
    distortion_frame.plot().legend(loc="upper left", bbox_to_anchor=(1,1))
    plt.xlabel('Iterations')
    plt.ylabel('Distortion Sum')
    plt.title('Kmeans Distortion')
    distortion_frame.isnull().sum().sum()
    #### kmeans plus plus
    distortion_frame_plus=pd.DataFrame(index=range(20),columns=range(20))
    for i in range(20):
        print(i)
        centriod_dict, distortion_sum=kmeans(mnist,k=10,max_iteration=20)
        distortion_frame_plus[i]=distortion_sum+[np.nan]*(20-len(distortion_sum))        
    distortion_frame_plus.plot().legend(loc="upper left", bbox_to_anchor=(1,1))
    plt.xlabel('Iterations')
    plt.ylabel('Distortion Sum')
    plt.title('Kmeans Plus Plus Distortion')
    distortion_frame_plus.isnull().sum().sum()







































