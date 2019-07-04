import tensorflow as tf

def add_variables_summaries(grads_and_vars, step):
  """Add summaries for variables.

  Args:
    grads_and_vars: A list of (gradient, variable) pairs.
    step: Variable to use for summaries.
  """
  with tf.name_scope('summarize_vars'):
    for grad, var in grads_and_vars:
      if grad is not None:
        if isinstance(var, tf.IndexedSlices):
          var_values = var.values
        else:
          var_values = var
        var_name = var.name.replace(':', '_')
        tf.compat.v2.summary.histogram(
            name=var_name + '_value', data=var_values, step=step)
        tf.compat.v2.summary.scalar(
            name=var_name + '_value_norm',
            data=tf.linalg.global_norm([var_values]),
            step=step)


def add_gradients_summaries(grads_and_vars, step):
  """Add summaries to gradients.

  Args:
    grads_and_vars: A list of gradient to variable pairs (tuples).
    step: Variable to use for summaries.
  """
  with tf.name_scope('summarize_grads'):
    for grad, var in grads_and_vars:
      if grad is not None:
        if isinstance(grad, tf.IndexedSlices):
          grad_values = grad.values
        else:
          grad_values = grad
        var_name = var.name.replace(':', '_')
        tf.compat.v2.summary.histogram(
            name=var_name + '_gradient', data=grad_values, step=step)
        tf.compat.v2.summary.scalar(
            name=var_name + '_gradient_norm',
            data=tf.linalg.global_norm([grad_values]),
            step=step)
      # else:
      #   logging.info('Var %s has no gradient', var.name)

def generate_tensor_summaries(tag, tensor, step):
  """Generates various summaries of `tensor` such as histogram, max, min, etc.

  Args:
    tag: A namescope tag for the summaries.
    tensor: The tensor to generate summaries of.
    step: Variable to use for summaries.
  """
  with tf.name_scope(tag):
    tf.compat.v2.summary.histogram(name='histogram', data=tensor, step=step)
    tf.compat.v2.summary.scalar(
        name='mean', data=tf.reduce_mean(input_tensor=tensor), step=step)
    tf.compat.v2.summary.scalar(
        name='mean_abs',
        data=tf.reduce_mean(input_tensor=tf.abs(tensor)),
        step=step)
    tf.compat.v2.summary.scalar(
        name='max', data=tf.reduce_max(input_tensor=tensor), step=step)
    tf.compat.v2.summary.scalar(
        name='min', data=tf.reduce_min(input_tensor=tensor), step=step)

