from tensorflow.keras import backend as K
import tensorflow as tf

def bag_accuracy(y_true, y_pred):
    """Compute accuracy of one bag.
    Parameters
    ---------------------
    y_true : Tensor (N x 1)
        GroundTruth of bag.
    y_pred : Tensor (1 X 1)
        Prediction score of bag.
    Return
    ---------------------
    acc : Tensor (1 x 1)
        Accuracy of bag label prediction.
    """
    y_true = K.mean(y_true, axis=0, keepdims=False)
    y_pred = K.mean(y_pred, axis=0, keepdims=False)
    acc = K.mean(K.equal(y_true, K.round(y_pred)))
    return acc


def bag_loss(y_true, y_pred):
    """Compute binary crossentropy loss of predicting bag loss.
    Parameters
    ---------------------
    y_true : Tensor (N x 1)
        GroundTruth of bag.
    y_pred : Tensor (1 X 1)
        Prediction score of bag.
    Return
    ---------------------
    acc : Tensor (1 x 1)
        Binary Crossentropy loss of predicting bag label.
    """
    y_true = K.mean(y_true, axis=0, keepdims=False)
    y_pred = K.mean(y_pred, axis=0, keepdims=False)
    loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    return loss


def get_contrastive_loss(m_neg=1, m_pos=.2):
    '''
    Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''

    def contrastive_loss(y_true, y_pred):
        return K.mean(y_true * K.square(K.maximum(y_pred - m_pos, 0)) +
                      (1 - y_true) * K.square(K.maximum(m_neg - y_pred, 0)))
    return contrastive_loss


def euclidean_distance(vects):
    """
    Calculate euclidean distance between two vectors
    Parameters
    ----------
    vects: Tuple[np.ndarray,np.ndarray]

    Returns
    -------
    a Tensor (1 x 1) of the euclidean distance between the two vectors.
    """
    x, y = vects

    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)

    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def siamese_accuracy(y_true, y_pred):
    """
    Compute classification accuracy for the siamese network
    Parameters
    ----------
    y_true : Tensor (1 x 1)
        ground truth of an instance
    y_pred : Tensor (1 X 1)
        distance score of the siamese network

    Returns
    -------
    a Tensor (1 x 1) classification score generated using a fixed distance threshold.
    """

    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


