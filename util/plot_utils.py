import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import cv2
from tqdm import tqdm
from util.utils import get_stamp_from_log, get_log_path
from os.path import join

colors_per_class = {
    '0': [254, 202, 87],
    '1': [255, 107, 107],
    '2': [10, 189, 227],
    '3': [255, 159, 243],
    '4': [16, 172, 132],
    '5': [128, 80, 128],
    '6': [87, 101, 116],
    '7': [52, 31, 151],
    '8': [0, 0, 0],
    '9': [100, 100, 255],
}


def cluster_and_visualize_images(feature_vectors, images, n_clusters=3):
    """
    Clusters images based on their corresponding feature vectors and visualizes them using t-SNE in 2D.

    Parameters:
    - feature_vectors (numpy array): Corresponding feature vectors for each image.
    - images (numpy array): List of images.
    - n_clusters (int): Number of clusters (default is 3).
    - image_size (tuple): Size of each image thumbnail in the plot (default is (50, 50)).
    - frame_width (int): Width of the color frame around each image (default is 5).
    """
    # Standardizing the vectors before clustering
    scaler = StandardScaler()
    scaled_vectors = scaler.fit_transform(feature_vectors)

    # Clustering the vectors using KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled_vectors)

    visualize_tsne(images=images, features=scaled_vectors, labels=labels, plot_size=2000, max_image_size=300)


# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def scale_image(image, max_image_size):
    image_height, image_width, _ = image.shape

    scale = max(1, image_width / max_image_size, image_height / max_image_size)
    image_width = int(image_width / scale)
    image_height = int(image_height / scale)

    image = cv2.resize(image, (image_width, image_height))
    return image


def draw_rectangle_by_class(image, label):
    image_height, image_width, _ = image.shape

    # get the color corresponding to image class
    color = colors_per_class[str(label)]
    image = cv2.rectangle(image, (0, 0), (image_width - 1, image_height - 1), color=color, thickness=5)

    return image


def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    image_height, image_width, _ = image.shape

    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset

    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size * (1 - y)) + offset

    # knowing the image center, compute the coordinates of the top left and bottom right corner
    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)

    br_x = tl_x + image_width
    br_y = tl_y + image_height

    return tl_x, tl_y, br_x, br_y


def visualize_tsne_images(tx, ty, images, labels, plot_size=1000, max_image_size=100):
    """
    Visualizes the T-SNE plot with images as points.
    :param tx: (numpy array) x-coordinates of the T-SNE plot
    :param ty: (numpy array) y-coordinates of the T-SNE plot
    :param images: (numpy array) list of image paths
    :param labels: (numpy array) list of image labels
    :param plot_size: (int) size of the plot
    :param max_image_size: (int) maximum size of the image thumbnail
    """
    offset = max_image_size // 2
    image_centers_area_size = plot_size - 2 * offset

    tsne_plot = 255 * np.ones((plot_size, plot_size, 3), np.uint8)

    # now we'll put a small copy of every image to its corresponding T-SNE coordinate
    for image_path, label, x, y in tqdm(
            zip(images, labels, tx, ty),
            desc='Building the T-SNE plot',
            total=len(images)
    ):
        image = cv2.imread(image_path)

        # scale the image to put it to the plot
        image = scale_image(image, max_image_size)

        # draw a rectangle with a color corresponding to the image class
        image = draw_rectangle_by_class(image, label)

        # compute the coordinates of the image on the scaled plot visualization
        tl_x, tl_y, br_x, br_y = compute_plot_coordinates(image, x, y, image_centers_area_size, offset)

        # put the image to its TSNE coordinates using numpy subarray indices
        tsne_plot[tl_y:br_y, tl_x:br_x, :] = image

    plt.imshow(tsne_plot[:, :, ::-1])
    plt.xticks([])
    plt.yticks([])

    # Saving the T-SNE plot to the log directory folder
    plt.savefig(join(get_log_path(), f'{get_stamp_from_log()}_tsne_plot_k{len(np.unique(labels))}.png'))

    # Showing the plot
    plt.show()


def visualize_tsne_points(tx, ty, labels):
    """
    Visualizes the T-SNE plot with points colored by class.
    :param tx: (numpy array) x-coordinates of the T-SNE plot
    :param ty: (numpy array) y-coordinates of the T-SNE plot
    :param labels: (numpy array) list of image labels
    :return: None
    """
    # initialize matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if str(l) == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([colors_per_class[label][::-1]]).astype(np.float32) / 255

        # add a scatter plot with the correponding color and label
        ax.scatter(current_tx, current_ty, c=color, label=label)

    # build a legend using the labels we set previously
    ax.legend(loc='best')

    # finally, show the plot
    plt.show()


def visualize_tsne(images, features, labels, plot_size=1000, max_image_size=100):
    """
    Visualizes the T-SNE plot with images and points colored by class.
    :param images: (numpy array) list of image paths
    :param features: (numpy array) feature vectors corresponding to each image
    :param labels: (numpy array) list of image labels
    :param plot_size: (int) size of the plot
    :param max_image_size: (int) maximum size of the image thumbnail
    :return: None
    """
    # Calculating appropriate perplexity (must be less than the number of samples)
    perplexity = min(30, len(features) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity).fit_transform(features)

    # Extracting x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # Scaling and moving the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # Visusalizing the plot: samples as colored points
    visualize_tsne_points(tx, ty, labels)

    # Visualizing the plot: samples as images
    visualize_tsne_images(tx, ty, images, labels, plot_size=plot_size, max_image_size=max_image_size)


def plot_loss_func(sample_count, loss_vals, loss_fig_path):
    plt.figure()
    plt.plot(sample_count, loss_vals)
    plt.grid()
    plt.title('Camera Pose Loss')
    plt.xlabel('Number of samples')
    plt.ylabel('Loss')
    plt.savefig(loss_fig_path)
