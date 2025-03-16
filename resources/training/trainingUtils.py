# author: Michael H�ppe
# date: 11.11.2024
# project: resources/trainingUtils.py
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
import tensorflow as tf
@tf.function
def knowledge_distillation_loss(y_true, y_pred_student, y_pred_teacher, temperature=1.0, alpha=0.5):
    """
    Combines hard and soft losses for knowledge distillation.
    """
    # Hard loss: Categorical cross-entropy with true labels
    hard_loss = masked_loss(y_true, y_pred_student)

    # Soft loss: KL divergence with teacher's soft predictions
    y_pred_teacher_soft = tf.nn.softmax(y_pred_teacher / temperature)
    y_pred_student_soft = tf.nn.softmax(y_pred_student / temperature)
    soft_loss = tf.keras.losses.KLDivergence()(y_pred_teacher_soft, y_pred_student_soft) * (temperature ** 2)

    # Combine losses
    return alpha * hard_loss + (1 - alpha) * soft_loss, soft_loss

@tf.function
def train_step_with_distillation(student_model, teacher_model, x, y_true, optimizer, temperature=1.0, alpha=0.5,
                                 return_prediction: bool = False):
    """
    Single training step for student model with knowledge distillation.
    """
    # Get teacher predictions
    y_pred_teacher = teacher_model(x, training=False)

    # Compute loss and apply gradients
    with tf.GradientTape() as tape:
        y_pred_student = student_model(x, training=True)
        hard_loss, soft_loss = knowledge_distillation_loss(y_true, y_pred_student, y_pred_teacher,
                                                           temperature=temperature, alpha=alpha)
        loss = tf.reduce_mean(hard_loss)
        soft_loss = tf.reduce_mean(soft_loss)

    grads = tape.gradient(loss, student_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, student_model.trainable_variables))
    if return_prediction:
        return loss, soft_loss, y_pred_student
    else:
        return loss, soft_loss

@tf.function
def masked_loss(y_true, logits):
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

    loss = loss_fn(y_true, logits)
    mask = tf.cast(y_true != 0, loss.dtype)
    loss *= mask

    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

@tf.function
def masked_loss_decoder_only(y_true, logits, sep_token_id=4):
    """
    Computes the masked loss for a decoder-only model using sparse categorical cross-entropy.

    This function calculates the loss between the true labels and the predicted logits,
    while ignoring entries before the separator token (`[SEP]`). The loss is only computed
    for tokens that come after the separator token.

    :param y_true: A tensor of true labels with shape (batch_size, seq_length).
                   Labels should be of integer type and each label should be an integer
                   index corresponding to the class.
    :param logits: A tensor of predicted logits with shape (batch_size, seq_length, num_classes).
                   This should be the raw output from the model (not softmaxed).
    :param sep_token_id: The token ID of the separator token (`[SEP]`).
    :return: A scalar tensor representing the mean masked loss across the non-masked tokens.
    """
    # Compute the loss
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE
    )
    loss = loss_fn(y_true, logits)

    # Create a mask to ignore tokens before the separator
    mask = tf.cast(y_true != 0, loss.dtype)  # Ignore padding tokens
    sep_mask = tf.cast(y_true == sep_token_id, tf.int32)  # Find separator tokens
    print(y_true, sep_mask)
    # Cumulative sum to identify positions after the separator
    sep_mask_cumsum = tf.cumsum(sep_mask, axis=0)
    causal_mask = tf.cast(sep_mask_cumsum > 0, loss.dtype)  # Mask for tokens after separator

    # Combine masks (ignore padding and tokens before separator)
    final_mask = mask * causal_mask

    # Apply the mask to the loss
    loss *= final_mask

    # Compute the mean loss over non-masked tokens
    return tf.reduce_sum(loss) / tf.reduce_sum(final_mask)

# @tf.function
def distillation_loss(y_true, y_pred, alpha=0.5, temperature=1.0):
    """
    Computes the distillation loss.
    Args:
        y_true: True labels.
        y_pred: Student model predictions.
        teacher_pred: Teacher model predictions.
        alpha: Weight for hard vs. soft loss.
        temperature: Temperature for softening probabilities.
    """
    # Hard loss (e.g., sparse categorical crossentropy)
    # hard_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    # hard_loss = masked_loss(y_true, y_pred)

    # Soft loss: KL divergence with teacher's soft predictions
    y_pred_teacher_soft = tf.nn.softmax(y_true / temperature)
    y_pred_student_soft = tf.nn.softmax(y_pred / temperature)
    soft_loss = tf.keras.losses.KLDivergence()(y_pred_teacher_soft, y_pred_student_soft) * (temperature ** 2)
    # Combine losses
    return soft_loss

# @tf.function
def masked_accuracy(y_true, y_pred):
    """
    Computes the masked accuracy for a classification task.

    This function calculates the accuracy of the predicted classes against the true labels,
    while ignoring entries where the true label is zero. The zero labels can represent
    padding or irrelevant data in the context of a sequence or batch.

    :param labels: A tensor of true labels with shape (batch_size,).
                            Labels should be of integer type and each label should
                            be an integer index corresponding to the class.
    :param y_pred: A tensor of predicted logits or probabilities with shape (batch_size, num_classes).
                            This can be the output from the model after applying a softmax
                            activation or the raw logits.
    :return:
    """
    # add teacher loss
    y_pred = tf.cast(tf.argmax(y_pred, axis=-1), y_true.dtype)
    mask = tf.cast(y_true != 0, tf.float32)
    accuracy = tf.cast(y_true == y_pred, tf.float32)
    accuracy *= mask

    return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

@tf.function
def masked_accuracy_decoder_only(y_true, y_pred, sep_token_id=102):
    """
    Computes the masked accuracy for a decoder-only model.

    This function calculates the accuracy of the predicted classes against the true labels,
    while ignoring entries before the separator token (`[SEP]`). The accuracy is only computed
    for tokens that come after the separator token.

    :param y_true: A tensor of true labels with shape (batch_size, seq_length).
                   Labels should be of integer type and each label should be an integer
                   index corresponding to the class.
    :param y_pred: A tensor of predicted logits or probabilities with shape (batch_size, seq_length, num_classes).
                   This can be the output from the model after applying a softmax activation or the raw logits.
    :param sep_token_id: The token ID of the separator token (`[SEP]`).
    :return: A scalar tensor representing the mean masked accuracy across the non-masked tokens.
    """
    # Convert predicted logits to class indices
    y_pred = tf.cast(tf.argmax(y_pred, axis=-1), y_true.dtype)

    # Create a mask to ignore tokens before the separator
    mask = tf.cast(y_true != 0, tf.float32)  # Ignore padding tokens
    sep_mask = tf.cast(y_true == sep_token_id, tf.int32)  # Find separator tokens

    # Cumulative sum to identify positions after the separator
    sep_mask_cumsum = tf.cumsum(sep_mask, axis=-1)
    causal_mask = tf.cast(sep_mask_cumsum > 0, tf.float32)  # Mask for tokens after separator

    # Combine masks (ignore padding and tokens before separator)
    final_mask = mask * causal_mask

    # Compute accuracy
    accuracy = tf.cast(y_true == y_pred, tf.float32)
    accuracy *= final_mask

    # Compute the mean accuracy over non-masked tokens
    return tf.reduce_sum(accuracy) / tf.reduce_sum(final_mask)

@register_keras_serializable()
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, embedding_dim, warmup_steps=4000):
        super().__init__()
        self.embedding_dim = tf.cast(embedding_dim, tf.float32)
        self.warmup_steps = warmup_steps
        step = tf.cast(1, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        self.learning_rate = tf.math.rsqrt(self.embedding_dim) * tf.math.minimum(arg1, arg2)

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        self.learning_rate = tf.math.rsqrt(self.embedding_dim) * tf.math.minimum(arg1, arg2)
        return self.learning_rate

    def get_config(self):
        return {
            "embedding_dim": self.embedding_dim.numpy(),
            "warmup_steps": self.warmup_steps
        }

