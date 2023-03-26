import cv2, numpy as np
from sklearn.cluster import KMeans

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

# Load image and convert to a list of pixels
image = cv2.imread('source_imgs/hello_garlic.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
reshape = image.reshape((image.shape[0] * image.shape[1], 3))

# Find and display most dominant colors
cluster = KMeans(n_clusters=5).fit(reshape)

visualize = visualize_colors(cluster, cluster.cluster_centers_)
visualize = cv2.cvtColor(visualize, cv2.COLOR_RGB2BGR)
cv2.imwrite('palettes/color_palette.png', visualize)
