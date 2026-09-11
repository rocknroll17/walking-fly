"""Compatibility shims for library version skew (import for side effects)."""
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def _device_put_replicated(x, devices):
    """Drop-in for the removed jax.device_put_replicated (official migration recipe)."""
    sharding = NamedSharding(Mesh(np.array(devices), ("x",)), P("x"))
    return jax.tree.map(lambda y: jax.device_put(jnp.stack([y] * len(devices)), sharding), x)


if not hasattr(jax, "device_put_replicated") or "deprecated" in (getattr(jax.device_put_replicated, "__doc__", "") or ""):
    jax.device_put_replicated = _device_put_replicated
