"""Smooth (differentiable) replacements for kinks, so the vector field stays C^1."""

import jax
import jax.numpy as jnp


def softplus(x, scale):
    """~max(x, 0), rounded over a width `scale`."""
    return scale * jax.nn.softplus(x / scale)


def soft_clip(x, lo, hi, scale):
    """~clip(x, lo, hi), rounded over a width `scale`."""
    return lo + softplus(x - lo, scale) - softplus(x - hi, scale)


def smoothstep(x):
    """0 for x<=0, 1 for x>=1, C^1 cubic in between."""
    x = jnp.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def safe_norm(v, eps=1e-9):
    """Euclidean norm over the last axis with a finite gradient at zero."""
    return jnp.sqrt(jnp.sum(v * v, axis=-1) + eps)


def perp(v):
    """Rotate a 2-vector (or array of them) by +90 deg: (x, z) -> (-z, x)."""
    return jnp.stack([-v[..., 1], v[..., 0]], axis=-1)


def cross2(r, f):
    """z-component of r x f (positive = nose-up moment in the x-z plane)."""
    return r[..., 0] * f[..., 1] - r[..., 1] * f[..., 0]


def rotate(theta, v):
    """Body -> earth rotation of a 2-vector by the pitch angle."""
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)
