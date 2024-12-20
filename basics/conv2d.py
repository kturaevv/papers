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


def conv2d_naive(x: jax.Array, kernel: jax.Array, stride: int = 1, padding: int = 0):
    assert kernel.ndim == 2, "Kernel should be 2D"
    assert kernel.shape[0] == kernel.shape[1], "Only rectangular kernels accepted"
    ksize = kernel.shape[0]
    padded_x = jax.numpy.pad(
        x, ((padding, padding), (padding, padding)), mode="constant"
    )
    out_shape = (x.shape[0] - kernel.shape[0] + 2 * padding) + 1
    out = jax.numpy.zeros((out_shape, out_shape))
    for yi in range(ksize // 2, x.shape[0] - ksize // 2, stride):
        for xi in range(ksize // 2, x.shape[1] - ksize // 2, stride):
            window = padded_x[
                yi - ksize // 2 : yi + ksize // 2 + 1,
                xi - ksize // 2 : xi + ksize // 2 + 1,
            ]
            res = jax.numpy.dot(window.flatten(), kernel.flatten())
            out = out.at[yi - ksize // 2, xi - ksize // 2].set(res)
    return out


def conv2d_naive_vec(
    x: jax.Array, kernel: jax.Array, stride: int = 1, padding: int = 0
):
    assert kernel.ndim == 2, "Kernel should be 2D"
    x = jax.numpy.pad(x, ((padding, padding), (padding, padding)), mode="constant")
    # Out matrix dimensions
    y_out = ((x.shape[0] - kernel.shape[0] + 2 * padding) // stride) + 1
    x_out = ((x.shape[1] - kernel.shape[1] + 2 * padding) // stride) + 1
    # Get meshgrid for static indexing
    out_y, out_x = jax.numpy.meshgrid(jax.numpy.arange(y_out), jax.numpy.arange(x_out))
    k_y, k_x = jax.numpy.meshgrid(
        jax.numpy.arange(kernel.shape[0]), jax.numpy.arange(kernel.shape[1])
    )
    # Get window with broadcasting (y_out, x_out, 1, 1) + (kernel.y, kernel.x) -> (y, x, ky, kx)
    window_y = (out_y[..., None, None] + k_y) * stride
    window_x = (out_x[..., None, None] + k_x) * stride
    # Index into x with matrices, basically creating a matrix of windows
    windows = x[window_y, window_x]
    # y, x, ky, kx . 1, 1, ky, kx
    windows_flat = windows.reshape(-1, kernel.shape[0] * kernel.shape[1])
    kernel_flat = kernel.flatten()
    return jax.numpy.dot(windows_flat, kernel_flat).reshape(y_out, x_out)


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

    arr2d = jax.random.normal(keys(), (50, 50))
    barr2d = jax.random.normal(keys(), (1, 50, 50))
    kernel = jax.numpy.ones((3, 3))
    bkernel = jax.numpy.ones((1, 3, 3))

    conv2d_naive_vec_jit = jax.jit(conv2d_naive_vec)
    conv2d_naive_vec_jit_vmap = jax.jit(jax.vmap(conv2d_naive_vec))
    conv2d_naive_jit = jax.jit(conv2d_naive)
    conv2d_naive_jit_vmap = jax.jit(jax.vmap(conv2d_naive))

    # %timeit conv2d_naive(arr2d, kernel)
    # %timeit conv2d_naive_jit(arr2d, kernel)
    # %timeit conv2d_naive_jit_vmap(barr2d, bkernel)
    # %timeit conv2d_naive_vec(arr2d, kernel)
    # %timeit conv2d_naive_vec_jit(arr2d, kernel)
    # %timeit conv2d_naive_vec_jit_vmap(barr2d, bkernel)
