import jax
from typing import Tuple, List


class RNNParams:
    def __init__(self, input_size: int, hidden_size: int):
        # Xavier initialization
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)

        scale_wx = jax.numpy.sqrt(2.0 / (input_size + hidden_size))
        scale_wh = jax.numpy.sqrt(2.0 / (hidden_size + hidden_size))
        scale_wy = jax.numpy.sqrt(2.0 / (hidden_size + 27))  # 27 output classes

        self.W_xh = jax.random.normal(k1, (input_size, hidden_size)) * scale_wx
        self.W_hh = jax.random.normal(k2, (hidden_size, hidden_size)) * scale_wh
        self.W_hy = jax.random.normal(k3, (hidden_size, 27)) * scale_wy

        self.b_h = jax.numpy.zeros(hidden_size)
        self.b_y = jax.numpy.zeros(27)

    def get_params(self):
        return [self.W_xh, self.W_hh, self.W_hy, self.b_h, self.b_y]


def create_char_mappings():
    chars = "abcdefghijklmnopqrstuvwxyz "  # 26 letters + space
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for idx, char in enumerate(chars)}
    return char_to_idx, idx_to_char


def one_hot_encode(char: str, char_to_idx: dict) -> jax.numpy.ndarray:
    idx = char_to_idx[char]
    return jax.numpy.eye(27)[idx]


def prepare_sequences(
    text: str, char_to_idx: dict
) -> Tuple[jax.numpy.ndarray, jax.numpy.ndarray]:
    """Prepare input-output sequences from text."""
    X = jax.numpy.stack([one_hot_encode(c, char_to_idx) for c in text[:-1]])
    y = jax.numpy.array([char_to_idx[c] for c in text[1:]])
    return X, y


@jax.jit
def rnn_forward(
    params: List[jax.numpy.ndarray], x: jax.numpy.ndarray, h: jax.numpy.ndarray
) -> Tuple[jax.numpy.ndarray, jax.numpy.ndarray]:
    """Single step forward pass of RNN."""
    W_xh, W_hh, W_hy, b_h, b_y = params
    h_new = jax.numpy.tanh(x @ W_xh + h @ W_hh + b_h)
    y = h_new @ W_hy + b_y
    return y, h_new


@jax.jit
def sequence_loss(
    params: List[jax.numpy.ndarray],
    inputs: jax.numpy.ndarray,
    targets: jax.numpy.ndarray,
) -> jax.numpy.ndarray:
    """Compute loss for a sequence."""

    def step(carry, x):
        h = carry
        y_pred, h_new = rnn_forward(params, x, h)
        return h_new, y_pred

    h_0 = jax.numpy.zeros(params[1].shape[0])  # Hidden state size from W_hh
    _, predictions = jax.lax.scan(step, h_0, inputs)

    # Compute cross entropy loss
    log_probs = jax.nn.log_softmax(predictions, axis=-1)
    loss = -jax.numpy.mean(log_probs[jax.numpy.arange(len(targets)), targets])
    return loss


@jax.jit
def update_params(
    params: List[jax.numpy.ndarray],
    grads: List[jax.numpy.ndarray],
    learning_rate: float,
) -> List[jax.numpy.ndarray]:
    """Update parameters using gradients."""
    return [p - learning_rate * g for p, g in zip(params, grads)]


def train_rnn(
    text: str, hidden_size: int = 64, learning_rate: float = 0.01, epochs: int = 1000
):
    """Train the RNN on the given text."""
    char_to_idx, idx_to_char = create_char_mappings()
    x, y = prepare_sequences(text, char_to_idx)

    init = RNNParams(27, hidden_size)  # 27 input size (one-hot encoding size)
    params = init.get_params()

    # Training loop
    grad_fn = jax.jit(jax.grad(sequence_loss))
    losses = []

    for epoch in range(epochs):
        grads = grad_fn(params, x, y)
        params = update_params(params, grads, learning_rate)

        if epoch % 100 == 0:
            loss = sequence_loss(params, x, y)
            losses.append(loss)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return params, losses, char_to_idx, idx_to_char


def generate_text(
    params: List[jax.numpy.ndarray],
    seed_char: str,
    length: int,
    char_to_idx: dict,
    idx_to_char: dict,
) -> str:
    """Generate text starting from a seed character."""
    h = jax.numpy.zeros(params[1].shape[0])
    generated_text = seed_char

    for _ in range(length):
        x = one_hot_encode(generated_text[-1], char_to_idx)
        y_pred, h = rnn_forward(params, x, h)
        next_char_idx = jax.numpy.argmax(y_pred)
        next_char = idx_to_char[int(next_char_idx)]
        generated_text += next_char

    return generated_text


if __name__ == "__main__":
    # Training and text generation
    data = "predictthissentencemyboi"
    params, losses, char_to_idx, idx_to_char = train_rnn(data, epochs=1000)

    # Generate some text
    seed_char = "p"
    generated = generate_text(
        params, seed_char, len(data) - 1, char_to_idx, idx_to_char
    )
    print("Generated text:", generated)
