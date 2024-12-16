# author: Michael Hüppe
# date: 11.11.2024
# project: resources/trainingUtils.py
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable


def masked_loss(labels, logits):
    """
    Computes the masked loss for a classification task using sparse categorical cross-entropy.

    This function calculates the loss between the true labels and the predicted logits,
    while ignoring entries where the label is zero. The zero labels can be used to
    represent padding or irrelevant data in the context of a sequence or batch.

    :param labels: A tensor of true labels with shape (batch_size,).
                            Labels should be of integer type and each label should
                            be an integer index corresponding to the class.
    :param logits: A tensor of predicted logits with shape (batch_size, num_classes).
                            This should be the raw output from the model (not softmaxed).
    :return: A scalar tensor representing the mean masked loss across the non-zero labels.
                   The loss is computed as the average of the masked losses, excluding zero labels.
    """
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE
    )

    loss = loss_fn(labels, logits)
    mask = tf.cast(labels != 0, loss.dtype)
    loss *= mask

    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def masked_accuracy(y_true, y_pred):
    """
    Computes the masked accuracy for a classification task.

    This function calculates the accuracy of the predicted classes against the true labels,
    while ignoring entries where the true label is zero. The zero labels can represent
    padding or irrelevant data in the context of a sequence or batch.

    :param y_true: A tensor of true labels with shape (batch_size,).
                            Labels should be of integer type and each label should
                            be an integer index corresponding to the class.
    :param y_pred: A tensor of predicted logits or probabilities with shape (batch_size, num_classes).
                            This can be the output from the model after applying a softmax
                            activation or the raw logits.
    :return:
    """
    y_pred = tf.cast(tf.argmax(y_pred, axis=-1), y_true.dtype)
    mask = tf.cast(y_true != 0, tf.float32)
    accuracy = tf.cast(y_true == y_pred, tf.float32)
    accuracy *= mask

    return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)


@register_keras_serializable()
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, embedding_dim, warmup_steps=4000):
        super().__init__()
        self.embedding_dim = tf.cast(embedding_dim, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.embedding_dim) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            "embedding_dim": self.embedding_dim.numpy(),
            "warmup_steps": self.warmup_steps
        }

