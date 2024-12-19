import jax
import jax.numpy as jnp


if __name__ == "__main__":
    key = jax.random.key(0)
    # What is a Neural network?
    # Basically a set of "trainable" weights
    key, W_key, b_key = jax.random.split(key, 3)
    W = jax.random.normal(W_key, (3,))
    b = jax.random.normal(b_key, ())

    # Which are processed along the inputs by also adding "non-linearity" which makes them "learn"
    sigmoid = lambda x: 0.5 * (jnp.tanh(x / 2) + 1)
    model = lambda W, b, inputs: jax.nn.sigmoid(jnp.dot(inputs, W) + b)

    # How much NN should "learn" is defined by a loss function
    def nll_loss(W, b):
        preds = model(W, b, inputs)
        label_probs = preds * targets + (1 - preds) * (1 - targets)
        return -jnp.sum(jnp.log(label_probs))

    # Toy dataset to calculate whether mean > or < 0
    inputs = jax.random.normal(key, (10, 3))
    targets = jnp.where(inputs.mean(axis=1) > 0, 1, 0)

    EPOCHS = 10
    for _ in range(EPOCHS):
        l = nll_loss(W, b)
        print(l)
        # Calculate gradients given input and weights with backpropagation
        dw, db = jax.grad(nll_loss, (0, 1))(W, b)
        # Update weights, i.e. make NN "learn"
        W -= dw
        b -= db
