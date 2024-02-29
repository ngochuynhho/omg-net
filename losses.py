import numpy as np
import tensorflow as tf
from tensorflow .keras import backend as K
from sklearn.metrics import accuracy_score, balanced_accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from skimage.metrics import structural_similarity
from typing import Any, Dict, Iterable, Sequence, Tuple, Optional, Union

cfce_func = tf.keras.losses.CategoricalFocalCrossentropy(alpha=0.1, gamma=4.)
cce_func = tf.keras.losses.CategoricalCrossentropy()
bce_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mse_func = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

def custom_focal_loss(y_true, y_pred):
    loss = cfce_func(y_true, y_pred)
    return loss

def custom_focal_loss_v2(y_true, y_pred, alpha=0.65, gamma=4):
    
    # Compute crossentropy loss
    ce_loss = cce_func(y_true, y_pred)
    
    # Compute focal loss
    pt = tf.exp(-ce_loss)
    focal_loss = alpha * tf.pow(1.0 - pt, gamma) * ce_loss
    
    # Mask out NaN losses
    valid_losses = tf.where(tf.math.is_finite(focal_loss), focal_loss, tf.zeros_like(focal_loss))
    
    # Take the mean of valid losses
    mean_loss = tf.reduce_mean(valid_losses)
    
    return mean_loss

def meam_squred_error_loss(y_true, y_pred):
    loss = mse_func(y_true, y_pred)
    return tf.reduce_mean(loss)

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)

def elbo_loss(x_true, x_pred, z, mean, logvar, mode='sum'):
    ce = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_pred, labels=x_true)
    if mode=='sum':
        logpx_z = -tf.reduce_sum(ce, axis=[1, 2, 3, 4])
    elif mode=='mean':
        logpx_z = -tf.reduce_mean(ce, axis=[1, 2, 3, 4])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

def kld_loss(x_true, x_pred, mean, logvar):
    mse_loss = mse_func(x_true, x_pred)
    kl_loss = -0.5 * K.sum(1 + logvar - K.square(mean) - K.exp(logvar), axis=-1)
    return tf.reduce_mean(mse_loss) + tf.reduce_mean(kl_loss)

def bce_one_var_loss(y_logits):
    loss = bce_func(tf.ones_like(y_logits), y_logits)
    return loss

def bce_two_var_loss(y_true, y_logits):
    true_loss = bce_func(tf.ones_like(y_true), y_true)
    logit_loss = bce_func(tf.zeros_like(y_logits), y_logits)
    return true_loss+logit_loss

class ContrastiveLoss:
    def __init__(
        self, feature_dimensions, temperature, queue_size,
    ):
        super().__init__()
        self.temperature = temperature
        self.feature_queue = tf.Variable(
            tf.math.l2_normalize(
                tf.random.normal(shape=(queue_size, feature_dimensions)), axis=1
            ),
            trainable=False,
        )

    def nearest_neighbour(self, projections):
        support_similarities = tf.matmul(
            projections, self.feature_queue, transpose_b=True
        )
        nn_projections = tf.gather(
            self.feature_queue, tf.argmax(support_similarities, axis=1), axis=0
        )
        return projections + tf.stop_gradient(nn_projections - projections)
    
    def contrastive_loss(self, projections_1, projections_2):
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)

        similarities_1_2_1 = (
            tf.matmul(
                self.nearest_neighbour(projections_1), projections_2, transpose_b=True
            )
            / self.temperature
        )
        similarities_1_2_2 = (
            tf.matmul(
                projections_2, self.nearest_neighbour(projections_1), transpose_b=True
            )
            / self.temperature
        )

        similarities_2_1_1 = (
            tf.matmul(
                self.nearest_neighbour(projections_2), projections_1, transpose_b=True
            )
            / self.temperature
        )
        similarities_2_1_2 = (
            tf.matmul(
                projections_1, self.nearest_neighbour(projections_2), transpose_b=True
            )
            / self.temperature
        )

        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            tf.concat(
                [
                    contrastive_labels,
                    contrastive_labels,
                    contrastive_labels,
                    contrastive_labels,
                ],
                axis=0,
            ),
            tf.concat(
                [
                    similarities_1_2_1,
                    similarities_1_2_2,
                    similarities_2_1_1,
                    similarities_2_1_2,
                ],
                axis=0,
            ),
            from_logits=True,
        )

        self.feature_queue.assign(
            tf.concat([projections_1, self.feature_queue[:-batch_size]], axis=0)
        )
        return loss

############################# Evaluation Metric ###############################

class CustomMetricS1:
    def reset_states(self) -> None:
        self._data = {
            "class_labels"      : [],
            "class_predictions" : [],
        }
    
    def update_state(self, y_true: Dict[str, tf.Tensor], y_pred: Dict[str, tf.Tensor]) -> None:
        self._data["class_labels"].append(y_true['conv_lb'].numpy())
        self._data["class_predictions"].append(y_pred.numpy())
    
    def _compute_classification_metric(self, y_true, y_pred) -> Dict[str, float]:
        y_true_argmax = np.argmax(y_true, axis=-1)
        y_pred_argmax = np.argmax(y_pred, axis=-1)
        results = {
            "accuracy" : accuracy_score(y_true_argmax, y_pred_argmax),
            "balance_accuracy" : balanced_accuracy_score(y_true_argmax, y_pred_argmax),
            "average_precision" : average_precision_score(y_true, y_pred, average="macro"),
            "f1" : f1_score(y_true_argmax, y_pred_argmax, average="macro"),
            "precision" : precision_score(y_true_argmax, y_pred_argmax, average="macro"),
            "recall"    : recall_score(y_true_argmax, y_pred_argmax, average="macro"),
            "roc_auc"   : roc_auc_score(y_true, y_pred, average="macro"),
        }
        return results
    
    def result(self) -> Dict[str, float]:
        data = {}
        for k, v in self._data.items():
            data[k] = np.concatenate(v)
        
        results = self._compute_classification_metric(
            data["class_labels"],
            data["class_predictions"])
        
        return results

def psnr(real_images, generated_images):
    mse = tf.reduce_mean(tf.square(real_images - generated_images))
    max_val = tf.reduce_max(real_images)
    psnr_value = 20 * tf.math.log(max_val / tf.sqrt(mse)) / tf.math.log(10.0)
    return psnr_value

def ssim(real_images, generated_images):
    # Convert to numpy arrays as structural_similarity expects numpy arrays
    real_np = np.squeeze(tf.keras.backend.eval(real_images))
    generated_np = np.squeeze(tf.keras.backend.eval(generated_images))
    
    ssim_value, _ = structural_similarity(real_np, generated_np,
                                          data_range=generated_np.max() - generated_np.min(),
                                          channel_axis=-1,
                                          full=True)
    return ssim_value