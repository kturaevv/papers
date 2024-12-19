import jax
from typing import Callable


def key_manager(key) -> Callable:
    state = {"idx": 0, "keys": jax.random.split(jax.random.key(key), 100)}

    def fn():
        state["idx"] += 1
        return state["keys"][state["idx"]]

    return fn


@jax.jit
def conv1d_naive(
    x: jax.Array, kernel: jax.Array, stride: int = 1, padding: int = 0
) -> jax.Array:
    assert kernel.ndim == 1, "Kernel should be 1D"
    ksize = kernel.shape[0]
    padded_x = jax.numpy.pad(x, (padding, padding), mode="constant")
    out = []
    for center in range(ksize // 2, x.size - ksize // 2, stride):
        window = padded_x[center - ksize // 2 : center + ksize // 2 + 1]
        out.append(jax.numpy.dot(window, kernel))
    return jax.numpy.array(out)


if __name__ == "__main__":
    keys = key_manager(0)

    arr = jax.random.normal(keys(), (3, 3, 3))
    arr, arr[:, 1, 0]

    arr = jax.random.normal(keys(), (1000))
    window = jax.numpy.array([3, 2, 1, 2, 3])
    assert jax.numpy.allclose(
        conv1d_naive(arr, window),
        jax.numpy.convolve(arr, window, mode="valid"),
        atol=1e-6,
    )
