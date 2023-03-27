"""
This code creates a palette for each image in a folder (source_imgs) and saves
it in a new folder called 'palettes'. The palettes are saved as palette_i.png
files where i is the number of the image in the folder. (This is useful for me
but the code can certainly be changed to save the palettes with the same name as
the original image.)

The code uses the KMeans algorithm from scikit-learn to cluster the colors in the
image. The number of clusters is set to 5, but this can be changed. The code also
uses OpenCV to read the images and create the palettes.
"""

#####################
## IMPORT PACKAGES ##
#####################

import cv2, numpy as np
from sklearn.cluster import KMeans
import os

########################
## MANAGE DIRECTORIES ##
########################

# Setup source directory and get list of files
src_dir = 'source_imgs/'
files = os.listdir(src_dir)
files = sorted(files)
# Filter out non-png files
image_files = [f for f in files if f.endswith('.png')]
# Create folder 'palettes' if it doesn't exist
if not os.path.exists('palettes'):
    os.makedirs('palettes')

################################
## THE PALETTE MAKER FUNCTION ##
################################

def visualize_colors(cluster, centroids):
    # Get the number of different clusters, create histogram, and normalize
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins=labels)
    hist = hist.astype("float")
    hist /= hist.sum()

    # Create square palette and iterate through each cluster's color and percentage
    palette = np.zeros((100, 100, 3), dtype=np.uint8)
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)], reverse=True)
    start = 0
    for i, (percent, color) in enumerate(colors):
        end = start + (percent * 100)
        cv2.rectangle(palette, (0, int(start)), (100, int(end)), color.astype("uint8").tolist(), -1)
        start = end
    return palette

#####################
## CREATE PALETTES ##
#####################

# Loop through the image files and save the corresponding palettes
for i, image_file in enumerate(image_files):
    # Load image and convert to a list of pixels
    image = cv2.imread(os.path.join(src_dir, image_file))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    reshape = image.reshape((image.shape[0] * image.shape[1], 3))

    # Find and display most dominant colors
    cluster = KMeans(n_clusters=5).fit(reshape)

    visualize = visualize_colors(cluster, cluster.cluster_centers_)
    visualize = cv2.cvtColor(visualize, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'palettes/palette_{i+1}.png', visualize)
