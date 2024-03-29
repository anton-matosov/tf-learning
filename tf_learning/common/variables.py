import tensorflow as tf

def soft_variables_update(source_variables,
                          target_variables,
                          tau=1.0,
                          sort_variables_by_name=False):
  """Performs a soft/hard update of variables from the source to the target.

  For each variable v_t in target variables and its corresponding variable v_s
  in source variables, a soft update is:
  v_t = (1 - tau) * v_t + tau * v_s

  When tau is 1.0 (the default), then it does a hard update:
  v_t = v_s

  Args:
    source_variables: list of source variables.
    target_variables: list of target variables.
    tau: A float scalar in [0, 1]. When tau is 1.0 (the default), we do a hard
      update.
    sort_variables_by_name: A bool, when True would sort the variables by name
      before doing the update.

  Returns:
    An operation that updates target variables from source variables.
  Raises:
    ValueError: if tau is not in [0, 1].
  """
  if tau < 0 or tau > 1:
    raise ValueError('Input `tau` should be in [0, 1].')
  updates = []

  op_name = 'soft_variables_update'
  if tau == 0.0 or not source_variables or not target_variables:
    return tf.no_op(name=op_name)
  if sort_variables_by_name:
    source_variables = sorted(source_variables, key=lambda x: x.name)
    target_variables = sorted(target_variables, key=lambda x: x.name)
  for (v_s, v_t) in zip(source_variables, target_variables):
    v_t.shape.assert_is_compatible_with(v_s.shape)
    if tau == 1.0:
      update = v_t.assign(v_s)
    else:
      update = v_t.assign((1 - tau) * v_t + tau * v_s)
    updates.append(update)
  return tf.group(*updates, name=op_name)
