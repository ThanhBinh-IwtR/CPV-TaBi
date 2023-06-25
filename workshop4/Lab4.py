import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from skimage.transform import resize
import matplotlib.pyplot as plt
import math
from pylab import*
from typing import Any
import random
import sys

class Snack:
    def __init__(self, raw_image, threshold = 5):
        self.raw_image = raw_image
        self.threshold = threshold
        image_changed = self.change_type_image()
        self.snake(self.raw_image, image_changed, self.threshold)

    def change_type_image(self):
        image = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2GRAY)
        image = np.array(image, dtype=np.float64)
        return image

    def mat_math(self, img, intput, str):
        output = intput
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if str == "atan":
                    output[i, j] = math.atan(intput[i, j])
                if str == "sqrt":
                    output[i, j] = math.sqrt(intput[i, j])
        return output

    def CV(self, LSF, img, mu, nu, epison, step):
        Drc = (epison / math.pi) / (epison * epison + LSF * LSF)
        Hea = 0.5 * (1 + (2 / math.pi) * self.mat_math(img, LSF / epison, "atan"))
        Iy, Ix = np.gradient(LSF)
        s = self.mat_math(img, Ix * Ix + Iy * Iy, "sqrt")
        Nx = Ix / (s + 0.000001)
        Ny = Iy / (s + 0.000001)
        Mxx, Nxx = np.gradient(Nx)
        Nyy, Myy = np.gradient(Ny)
        cur = Nxx + Nyy
        Length = nu * Drc * cur

        Lap = cv2.Laplacian(LSF, -1)
        Penalty = mu * (Lap - cur)

        s1 = Hea * img
        s2 = (1 - Hea) * img
        s3 = 1 - Hea
        C1 = s1.sum() / Hea.sum()
        C2 = s2.sum() / s3.sum()
        CVterm = Drc * (-1 * (img - C1) * (img - C1) + 1 * (img - C2) * (img - C2))

        LSF = LSF + step * (Length + Penalty + CVterm)
        return LSF

    def snake(self, raw_image, image_vector, num):
        IniLSF = np.ones((image_vector.shape[0], image_vector.shape[1]), image_vector.dtype)
        IniLSF[30:80, 30:80] = -1
        IniLSF = -IniLSF
        mu = 1
        nu = 0.003 * 255 * 255
        epison = 1
        step = 0.1
        LSF = IniLSF

        image_with_contour = np.copy(raw_image)
        for _ in range(1, num):
            LSF = self.CV(LSF, image_vector, mu, nu, epison, step)
            contours, hierarchy = cv2.findContours(LSF.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_with_contour, contours, -1, (0, 0, 255), 2)
        cv2.imshow("Image with contour", image_with_contour)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# def watershed(img):
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#     # Noise removal
#     kernel = np.ones((2,2),np.uint8)
#     closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)
#     sure_bg = cv2.dilate(closing,kernel,iterations=3)
#     # Finding sure foreground area
#     dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,3)
#     # Threshold
#     ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)
#     # Finding unknown region
#     sure_fg = np.uint8(sure_fg)
#     unknown = cv2.subtract(sure_bg,sure_fg)
#     # Marker labelling
#     ret, markers = cv2.connectedComponents(sure_fg)
#     # Add one to all labels so that sure background is not 0, but 1
#     markers = markers+1
#     # Now, mark the region of unknown with zero
#     markers[unknown==255] = 0
#     markers = cv2.watershed(img,markers)
#     img[markers == -1] = [255,0,0]
#     unknown = unknown.reshape((unknown.shape[0], unknown.shape[1], 1))
#     cv2.imshow('Unknown', unknown)
#     cv2.imshow('Watershed', img)
#     cv2.waitKey(0)  # Wait for any key press
#     cv2.destroyAllWindows()  # Close all windows

def neighbourhood(image, x, y):
    # Save the neighbourhood pixel's values in a dictionary
    neighbour_region_numbers = {}
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i == 0 and j == 0):
                continue
            if (x+i < 0 or y+j < 0): # If coordinates out of image range, skip
                continue
            if (x+i >= image.shape[0] or y+j >= image.shape[1]): # If coordinates out of image range, skip
                continue
            if (neighbour_region_numbers.get(image[x+i][y+j]) == None):
                neighbour_region_numbers[image[x+i][y+j]] = 1 # Create entry in dictionary if not already present
            else:
                neighbour_region_numbers[image[x+i][y+j]] += 1 # Increase count in dictionary if already present

    # Remove the key - 0 if exists
    if (neighbour_region_numbers.get(0) != None):
        del neighbour_region_numbers[0]

    # Get the keys of the dictionary
    keys = list(neighbour_region_numbers)

    # Sort the keys for ease of checking
    keys.sort()

    if (keys[0] == -1):
        if (len(keys) == 1): # Separate region
            return -1
        elif (len(keys) == 2): # Part of another region
            return keys[1]
        else: # Watershed
            return 0
    else:
        if (len(keys) == 1): # Part of another region
            return keys[0]
        else: # Watershed
            return 0

def watershed_segmentation(image):
    # Create a list of pixel intensities along with their coordinates
    intensity_list = []
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            # Append the tuple (pixel_intensity, xy-coord) to the end of the list
            intensity_list.append((image[x][y], (x, y)))

    # Sort the list with respect to their pixel intensities, in ascending order
    intensity_list.sort()

    # Create an empty segmented_image numpy ndarray initialized to -1's
    segmented_image = np.full(image.shape, -1, dtype=int)

    # Iterate the intensity_list in ascending order and update the segmented image
    region_number = 0
    for i in range(len(intensity_list)):
        # Print iteration number in terminal for clarity

        # Get the pixel intensity and the x,y coordinates
        intensity = intensity_list[i][0]
        x = intensity_list[i][1][0]
        y = intensity_list[i][1][1]

        # Get the region number of the current pixel's region by checking its neighbouring pixels
        region_status = neighbourhood(segmented_image, x, y)

        # Assign region number (or) watershed accordingly, at pixel (x, y) of the segmented image
        if (region_status == -1): # Separate region
            region_number += 1
            segmented_image[x][y] = region_number
        elif (region_status == 0): # Watershed
            segmented_image[x][y] = 0
        else: # Part of another region
            segmented_image[x][y] = region_status

    # Return the segmented image
    cv2.imwrite("target.png", segmented_image)
    seg_image = cv2.resize(cv2.imread("target.png", 0), (0,0), None, 1, 1)
    cv2.imshow('original', image)
    cv2.imshow('new image', seg_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


np.random.seed(42)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KMeans():

    def __init__(self, image, K=5, max_iters=100, plot_steps=False):
        self.image = image
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # the centers (mean feature vector) for each cluster
        self.centroids = []

    def predict(self, X):
        self.X = X
        print(self.X.shape)
        self.n_samples, self.n_features = X.shape
        
        # initialize 
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # Optimize clusters
        for _ in range(self.max_iters):
            # Assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)
            if self.plot_steps:
                self.plot()

            # Calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            
            # check if clusters have changed
            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        # Classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)


    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        # Assign the samples to the closest centroids to create clusters
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def _get_centroids(self, clusters):
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        # distances between each old and new centroids, fol all centroids
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color='black', linewidth=2)

        plt.show()
    def cent(self):
        return self.centroids
    
    def run(self):
        pixel_values = self.image.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        k = KMeans(self.image, self.K, self.max_iters)
        y_pred = k.predict(pixel_values).astype(int)
        centers = np.uint8(k.cent())

        labels = y_pred.flatten()
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(self.image.shape)
        cv2.imshow('original image', self.image)
        cv2.imshow('new image', segmented_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def mean_shift_peak_detection(K):
    row = K.shape[0]
    col = K.shape[1]

    J = row * col
    Size = row, col, 3
    R = np.zeros(Size, dtype=np.uint8)
    D = np.zeros((J, 5))
    arr = np.array((1, 3))

    # cv2.imshow("image", K)

    counter = 0
    iter = 1.0

    threshold = 120
    current_mean_random = True
    current_mean_arr = np.zeros((1, 5))
    below_threshold_arr = []

    # converted the image K[rows][col] into a feature space D. The dimensions of D are [rows*col][5]
    for i in range(0, row):
        for j in range(0, col):
            arr = K[i][j]

            for k in range(0, 5):
                if (k >= 0) & (k <= 2):
                    D[counter][k] = arr[k]
                else:
                    if (k == 3):
                        D[counter][k] = i
                    else:
                        D[counter][k] = j
            counter += 1

    while (len(D) > 0):
        # print J
        # selecting a random row from the feature space and assigning it as the current mean
        if (current_mean_random):
            current_mean = random.randint(0, len(D) - 1)
            for i in range(0, 5):
                current_mean_arr[0][i] = D[current_mean][i]
        below_threshold_arr = []
        for i in range(0, len(D)):
            # print "Entered here"
            ecl_dist = 0
            color_total_current = 0
            color_total_new = 0
            # Finding the eucledian distance of the randomly selected row i.e. current mean with all the other rows
            for j in range(0, 5):
                ecl_dist += ((current_mean_arr[0][j] - D[i][j]) ** 2)

            ecl_dist = ecl_dist ** 0.5

            # Checking if the distance calculated is within the threshold. If yes taking those rows and adding
            # them to a list below_threshold_arr

            if (ecl_dist < threshold):
                below_threshold_arr.append(i)
                # print "came here"

        mean_R = 0
        mean_G = 0
        mean_B = 0
        mean_i = 0
        mean_j = 0
        current_mean = 0
        mean_col = 0

        # For all the rows found and placed in below_threshold_arr list, calculating the average of
        # Red, Green, Blue and index positions.

        for i in range(0, len(below_threshold_arr)):
            mean_R += D[below_threshold_arr[i]][0]
            mean_G += D[below_threshold_arr[i]][1]
            mean_B += D[below_threshold_arr[i]][2]
            mean_i += D[below_threshold_arr[i]][3]
            mean_j += D[below_threshold_arr[i]][4]

        mean_R = mean_R / len(below_threshold_arr)
        mean_G = mean_G / len(below_threshold_arr)
        mean_B = mean_B / len(below_threshold_arr)
        mean_i = mean_i / len(below_threshold_arr)
        mean_j = mean_j / len(below_threshold_arr)

        # Finding the distance of these average values with the current mean and comparing it with iter

        mean_e_distance = ((mean_R - current_mean_arr[0][0]) ** 2 + (mean_G - current_mean_arr[0][1]) ** 2 + (
                    mean_B - current_mean_arr[0][2]) ** 2 + (mean_i - current_mean_arr[0][3]) ** 2 + (
                                       mean_j - current_mean_arr[0][4]) ** 2)

        mean_e_distance = mean_e_distance ** 0.5

        nearest_i = 0
        min_e_dist = 0
        counter_threshold = 0
        # If less than iter, find the row in below_threshold_arr that has i,j nearest to mean_i and mean_j
        # This is because mean_i and mean_j could be decimal values which do not correspond
        # to actual pixel in the Image array.

        if (mean_e_distance < iter):
            # print "Entered here"
            '''    
            for i in range(0, len(below_threshold_arr)):
                new_e_dist = ((mean_i - D[below_threshold_arr[i]][3])**2 + (mean_j - D[below_threshold_arr[i]][4])**2 + (mean_R - D[below_threshold_arr[i]][0])**2 +(mean_G - D[below_threshold_arr[i]][1])**2 + (mean_B - D[below_threshold_arr[i]][3])**2)
                new_e_dist = new_e_dist**0.5
                if(i == 0):
                    min_e_dist = new_e_dist
                    nearest_i = i
                else:
                    if(new_e_dist < min_e_dist):
                        min_e_dist = new_e_dist
                        nearest_i = i
    '''
            new_arr = np.zeros((1, 3))
            new_arr[0][0] = mean_R
            new_arr[0][1] = mean_G
            new_arr[0][2] = mean_B

            # When found, color all the rows in below_threshold_arr with
            # the color of the row in below_threshold_arr that has i,j nearest to mean_i and mean_j
            for i in range(0, len(below_threshold_arr)):
                R[int(D[below_threshold_arr[i]][3])][int(D[below_threshold_arr[i]][4])] = new_arr

                # Also now don't use those rows that have been colored once.

                D[below_threshold_arr[i]][0] = -1
            current_mean_random = True
            new_D = np.zeros((len(D), 5))
            counter_i = 0

            for i in range(0, len(D)):
                if (D[i][0] != -1):
                    new_D[counter_i][0] = D[i][0]
                    new_D[counter_i][1] = D[i][1]
                    new_D[counter_i][2] = D[i][2]
                    new_D[counter_i][3] = D[i][3]
                    new_D[counter_i][4] = D[i][4]
                    counter_i += 1

            D = np.zeros((counter_i, 5))

            counter_i -= 1
            for i in range(0, counter_i):
                D[i][0] = new_D[i][0]
                D[i][1] = new_D[i][1]
                D[i][2] = new_D[i][2]
                D[i][3] = new_D[i][3]
                D[i][4] = new_D[i][4]

        else:
            current_mean_random = False

            current_mean_arr[0][0] = mean_R
            current_mean_arr[0][1] = mean_G
            current_mean_arr[0][2] = mean_B
            current_mean_arr[0][3] = mean_i
            current_mean_arr[0][4] = mean_j

    cv2.imshow('before', K)
    cv2.imshow('after', R)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    print("Welcome to Image Processing Functions")
    print("Please select a function from the following options:")

    while True:
        image_path = 'road.jpg'
        image = cv2.imread(image_path)
        print("1. Snakes Algorithm for Active Contours")
        print("2. Watershed Algorithm for Image Segmentation")
        print("3. K-Means for Segmentation")
        print("4. Mean Shift for Peak Detection")
        print("0. Exit")
        choice = int(input("Enter your choice (0-4): "))

        if choice == 1:
            print("Snakes Algorithm for Active Contours selected.")
            snake = Snack(image)

        elif choice == 2:
            print("Watershed Algorithm for Image Segmentation selected.")
            # watershed(image)
            image_gray = cv2.imread(image_path, 0)
            watershed_segmentation(image_gray)

        elif choice == 3:
            print("K-Means for Segmentation selected.")
            KMeans(image, K=4, max_iters=10).run()

        elif choice == 4:
            print("Mean Shift for Peak Detection selected.")
            mean_shift_peak_detection(image)

        elif choice == 0:
            print("Thank you for using Image Processing Functions. Goodbye!")
            break

        else:
            print("Invalid choice. Please select again.")

if __name__ == '__main__':
    main()

