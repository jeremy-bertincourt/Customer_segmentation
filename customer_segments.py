#!/usr/bin/python2.7

# Import libraries necessary for this project
import numpy as np
import pandas as pd
import renders as rs
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.mixture import GMM
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"
    
# get the feature correlations
corr = data.corr()

# remove first row and last column for a cleaner look
corr.drop(['Fresh'], axis=0, inplace=True)
corr.drop(['Delicatessen'], axis=1, inplace=True)

# create a mask so we only see the correlation values once
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, 1)] = True

# plot the heatmap
with sns.axes_style("white"):
    sns.heatmap(corr, mask=mask, annot=True, cmap='RdBu', fmt='+.2f', cbar=False)
plt.show()
    
#Scale the data using the natural logarithm
log_data = np.log(data)

# Produce a scatter matrix for each pair of newly-transformed features
pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');
plt.show()

# For each feature find the data points with extreme high or low values
for feature in log_data.keys():
    
    # Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature], 25)
    
    # Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature], 75)

    # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = 1.5*(Q3 - Q1)
    
    # Display the outliers
    print "Data points considered outliers for the feature '{}':".format(feature)
    print(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])

# Select the indices for data points you wish to remove
outliers  = []

# Remove the outliers
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)

# Apply PCA by fitting the good data with the same number of dimensions as features
pca = PCA(n_components=2)
pca.fit(good_data)

# Transform the sample log-data using the PCA fit above
reduced_data = pca.transform(good_data)

# Generate PCA results plot
pca_results = rs.pca_results(good_data, pca)
plt.show()

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])
pd.scatter_matrix(reduced_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');
plt.show()

# Assess the silhouette coefficient value in order to know the best number of clusters to choose
scores = []
for x in range(2, 4):
    gmm = GMM(n_components=x)
    clusterer = gmm.fit(reduced_data)

    preds = clusterer.predict(reduced_data)

    centers = clusterer.means_ 

    score = silhouette_score(reduced_data, preds)

    scores.append(score)
print scores

# Display the results of the clustering from implementation
rs.cluster_results(reduced_data, preds, centers)
plt.show()

# Inverse transform the centers
log_centers = pca.inverse_transform(centers)

# Exponentiate the centers
true_centers = np.exp(log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments

true_centers = true_centers.append(data.describe().loc['50%'])
true_centers = true_centers.append(data.describe().loc['mean'])
true_centers.index = segments + ['median'] + ['mean']
true_centers.plot(kind='bar', figsize=(15,6))
plt.show()


