"""Smooth spring-damper ground contact (runway at z = 0)."""

import jax
import jax.numpy as jnp

from .smooth import softplus


def contact_force(penetration, vel, k, c, mu, v_slip=0.1):
    """Normal + friction force at contact point(s) with the ground.

    `penetration` = -z of the point (positive when below ground), `vel` its velocity.
    The normal force k*pen - c*vz is only allowed to push (softplus), and fades out
    smoothly above the ground; friction is Coulomb, regularised with tanh.
    """
    on_ground = jax.nn.sigmoid(penetration / 2e-3)
    N = on_ground * softplus(k * penetration - c * vel[..., 1], 1.0)
    Fx = -mu * N * jnp.tanh(vel[..., 0] / v_slip)
    return jnp.stack([Fx, N], axis=-1)
