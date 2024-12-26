import jax


def scaled_dot_product_attention(
    params: list[jax.Array], input: jax.Array, d_k: int
) -> jax.Array:
    W_q, W_k, W_v = params
    assert (
        W_q.ndim == 3
    ), f"Weights should be (batch_size, seq_len, emb_size), got {W_q.ndim}"
    assert (
        W_k.ndim == 3
    ), f"Weights should be (batch_size, seq_len, emb_size), got {W_k.ndim}"
    assert (
        W_v.ndim == 3
    ), f"Weights should be (batch_size, seq_len, emb_size), got {W_v.ndim}"

    # Input shape: (batch_size, seq_len, emb_size)
    # W_q, W_k, W_v shapes: (batch_size, emb_size, d_k/d_v)

    # Shape becomes (batch_size, seq_len, d_k)
    Q = input @ W_q
    K = input @ W_k
    V = input @ W_v

    # For the attention score, we want (batch_size, seq_len, seq_len)
    # Need to transpose K to (batch_size, d_k, seq_len)
    K = jax.numpy.transpose(K, (0, 2, 1))

    # (batch_size, seq_len, seq_len)
    scaled_attn = (Q @ K) / jax.numpy.sqrt(d_k)

    # Create mask of shape (seq_len, seq_len)
    mask = jax.numpy.tril(jax.numpy.ones((scaled_attn.shape[1], scaled_attn.shape[2])))
    masked_attn = jax.numpy.where(mask == 0, float("-inf"), scaled_attn)

    # Softmax along seq_len dimension
    norm_attn = jax.nn.softmax(masked_attn, axis=-1)

    # Final multiplication with V to get (batch_size, seq_len, d_v)
    return norm_attn @ V


def multi_head_attention(
    params: list[jax.Array], input: jax.Array, num_heads: int, d_model: int
) -> jax.Array:
    W_q, W_k, W_v, W_o = params

    batch_size, seq_len = input.shape[0:2]
    d_k = d_model // num_heads
    d_v = d_model // num_heads

    # Split heads: (batch, seq_len, num_heads, d_k)
    Q = (input @ W_q).reshape(batch_size, seq_len, num_heads, d_k)
    K = (input @ W_k).reshape(batch_size, seq_len, num_heads, d_k)
    V = (input @ W_v).reshape(batch_size, seq_len, num_heads, d_v)

    # Transpose: (batch, num_heads, seq_len, d_k)
    Q = jax.numpy.transpose(Q, (0, 2, 1, 3))
    K = jax.numpy.transpose(K, (0, 2, 1, 3))
    V = jax.numpy.transpose(V, (0, 2, 1, 3))

    # Scaled dot product for each head
    K_t = jax.numpy.transpose(K, (0, 1, 3, 2))
    scaled_attn = (Q @ K_t) / jax.numpy.sqrt(d_k)

    # Masking
    mask = jax.numpy.tril(jax.numpy.ones((seq_len, seq_len)))
    masked_attn = jax.numpy.where(mask == 0, float("-inf"), scaled_attn)
    norm_attn = jax.nn.softmax(masked_attn, axis=-1)

    # Attention output: (batch, num_heads, seq_len, d_v)
    head_output = norm_attn @ V

    # Concatenate heads and project
    concat = jax.numpy.transpose(head_output, (0, 2, 1, 3))
    concat = concat.reshape(batch_size, seq_len, d_model)

    return concat @ W_o


if __name__ == "__main__":
    key = jax.random.key(0)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    batch_size = 1
    seq_len = 10
    d_model = 128
    num_heads = 4
    d_k = 32  # hidden size
    d_v = 64  # value size

    # Input shape: (batch_size, seq_len, emb_size)
    input = jax.random.normal(k5, (batch_size, seq_len, d_model))

    # Shape: (batch_size, emb_size, d_k/d_v)
    W_q = jax.random.normal(k1, (batch_size, d_model, d_k))
    W_k = jax.random.normal(k2, (batch_size, d_model, d_k))
    W_v = jax.random.normal(k3, (batch_size, d_model, d_v))
    params = [W_q, W_k, W_v]

    self_attn = scaled_dot_product_attention(params, input, d_k)
    # Output shape should be (batch_size, seq_len, d_v)
    print("Scaled attention:", self_attn.shape)

    num_heads = 8

    W_q = jax.random.normal(k1, (batch_size, d_model, d_model))
    W_k = jax.random.normal(k2, (batch_size, d_model, d_model))
    W_v = jax.random.normal(k3, (batch_size, d_model, d_model))
    W_o = jax.random.normal(k4, (batch_size, d_model, d_model))
    params = [W_q, W_k, W_v, W_o]

    output = multi_head_attention(params, input, num_heads, d_model)
    # Should be (batch_size, seq_len, d_model)
    print(f"Multi-head attention: {output.shape}")
