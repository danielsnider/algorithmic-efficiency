"""Tests for imagenet_jax/workload.py."""

from absl.testing import absltest
import jax
import jax.numpy as jnp

from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.imagenet.imagenet_jax.workload import ImagenetJaxWorkload


def _pytree_total_diff(pytree_a, pytree_b):
  pytree_diff = jax.tree_map(lambda a, b: jnp.sum(a - b), pytree_a, pytree_b)
  pytree_diff = jax.tree_leaves(pytree_diff)
  return jnp.sum(jnp.array(pytree_diff))


class ModelsTest(absltest.TestCase):
  """Tests for imagenet_jax/workload.py."""

  def test_forward_pass(self):
    batch_size = 11
    rng = jax.random.PRNGKey(0)
    rng, model_init_rng, *data_rngs = jax.random.split(rng, 4)
    workload = ImagenetJaxWorkload()
    model_params, batch_stats = workload.init_model_fn(model_init_rng)
    input_shape = (jax.local_device_count(), batch_size, 224, 224, 3)
    first_input_batch = jax.random.normal(data_rngs[0], shape=input_shape)
    expected_logits_shape = (jax.local_device_count(), batch_size, 1000)

    pmapped_model_fn = jax.pmap(
        workload.model_fn,
        axis_name='batch',
        in_axes=(0, 0, 0, None, None, None),
        static_broadcasted_argnums=(3, 5))
    logits, updated_batch_stats = pmapped_model_fn(
        model_params,
        first_input_batch,
        batch_stats,
        spec.ForwardPassMode.TRAIN,
        rng,
        True)
    self.assertEqual(logits.shape, expected_logits_shape)
    # Test that batch stats are updated.
    self.assertNotEqual(
        _pytree_total_diff(batch_stats, updated_batch_stats), 0.0)

    second_input_batch = jax.random.normal(data_rngs[1], shape=input_shape)
    # Test that batch stats are not updated when we say so.
    _, same_batch_stats = pmapped_model_fn(
        model_params,
        second_input_batch,
        batch_stats,
        spec.ForwardPassMode.TRAIN,
        rng,
        False)
    self.assertIsNone(same_batch_stats)

    # Test eval model.
    logits, _ = pmapped_model_fn(
        model_params,
        second_input_batch,
        batch_stats,
        spec.ForwardPassMode.EVAL,
        rng,
        False)
    self.assertEqual(logits.shape, expected_logits_shape)


if __name__ == '__main__':
  absltest.main()
