import numpy as np
import sklearn

def extract_centers(matrices):
    '''Given the sequence of matrices where masks for fingers are extracted, compute
    the center of the masks for each different one. The center of the mask is a
    good point to estimate movement of the finger
    '''

    n_masks = matrices.max().astype(int)
    centers = np.zeros((n_masks, matrices.shape[0], 2)).astype(int)
    for value in range(1, n_masks+1):
        blob_coords = np.argwhere(matrices == value)
        for n_step in range(matrices.shape[0]):
            centers[value - 1, n_step] = blob_coords[blob_coords[:, 0] == n_step, 1:].mean(axis=0)
    return centers


def extract_extreme_points(matrices):
    '''Given the sequence of matrices where masks for fingers are extracted, compute
    the extreme points of each mask in each frame. Extreme points are defined as
    the center of each edge, if the mask is considered to be a rectangle. Using
    PCA, those points can be calculated using the principal components, the
    center of the mask, and the distance between the border of the mask and the
    center.
    '''

    # TODO: Adapt this code to work in general (for now it is cut from the
    # trials script, so the variables don't have a proper name and some of them
    # have to be computed). Also, this is done only for one mask, it should be
    # generalized for the number of masks contained in the matrices (usually 2))

    # PCA for finger
    first_finger = np.argwhere(matrices == 1)
    first_frame = first_finger[first_finger[:,0] == 0][:, 1:]
    center = centers[0,0]
    pca = sklearn.decomposition.PCA(2)
    pca.fit(first_frame)
    plt.imshow(masks[0])
    for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
        min_val = first_frame[:, 1-i].min()
        max_val = first_frame[:, 1-i].max()
        plt.plot(
            [center[1] + (min_val - center[1-i])*comp[1], center[1] + (max_val - center[1-i])*comp[1]],
            [center[0] + (min_val - center[1-i])*comp[0], center[0] + (max_val - center[1-i])*comp[0]],
            label=f"Component {i}",
            linewidth=1,
            color=f"C{i + 3}",
        )
    plt.show()
