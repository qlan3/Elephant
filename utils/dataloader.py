import jax.numpy as jnp


def load_data(dataset, seed=None, batch_size=1):
    dataset = dataset.lower()

    if dataset in ["sin", "streamsin"]:
        train_size, test_size = 200, 1000
        train_data, test_data = dict(), dict()
        train_data["x"] = jnp.linspace(0.0, 2.0, num=train_size).reshape((-1, 1))
        train_data["y"] = jnp.sin(jnp.pi * train_data["x"])
        test_data["x"] = jnp.linspace(0.0, 2.0, num=test_size).reshape((-1, 1))
        test_data["y"] = jnp.sin(jnp.pi * test_data["x"])
        dummy_input = jnp.ones([1, 1])

    data_loader = {"Train": train_data, "Test": test_data, "dummy_input": dummy_input}
    return data_loader
