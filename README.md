### Subject

This program clusters customers in order to aim at the most representative customers when changing the company strategy.

### How it works

The program first displays a heatmap of the correlation between the features. Then the features are scaled and the outliers are displayed. After applying PCA algorithm, only the 2 principal features are retained. In order to segment customer, Gaussian Mixture Model is used and the number of cluters to apply is determined with the silhouette coefficient. The 3 clusters are then displayed. Finally, the inverve transformation and the exponentional is applied to the centers in order to determine the most representative customers.

### How to run the program

run the following command once in the main directory:
python customer_segments.py 
