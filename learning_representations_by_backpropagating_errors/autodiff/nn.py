import random
from .model import Scalar


class Units:  # aka a single Neuron
    def __init__(self, n_in: int) -> None:
        self.w: list[Scalar] = [Scalar(random.uniform(-1, 1)) for _ in range(n_in)]
        self.bias: Scalar = Scalar(1.0)

    def __call__(self, y: list["Scalar"]) -> "Scalar":
        x_j: Scalar = sum((y_i * w_ji for y_i, w_ji in zip(y, self.w)), self.bias)
        out = x_j.tanh()
        return out

    def __repr__(self) -> str:
        doc: str = "Unit[\n"
        for x_i in self.w:
            doc += "\t" + x_i.__str__() + "\n"
        doc += "]"
        return doc

    def parameters(self):
        return self.w + [self.bias]


class StateVector:  # aka Tensor
    def __init__(self, n_in: int, n_out: int) -> None:
        self.n_in: int = n_in
        self.n_out: int = n_out
        self.units: list[Units] = [Units(n_in) for _ in range(n_out)]

    def __call__(self, y: list["Scalar"]) -> list[Scalar]:
        out: list[Scalar] = [unit(y) for unit in self.units]
        return out

    def __repr__(self) -> str:
        doc: str = f"StateVector[{self.n_in}, {self.n_out}][\n"
        for j in self.units:
            doc += "  " + j.__str__() + "\n"
        doc += "]"
        return doc

    def parameters(self):
        return [weights for unit in self.units for weights in unit.parameters()]


class NeuralNetwork:
    def __init__(self, layers: list[StateVector]) -> None:
        self.layers: list[StateVector] = layers

    def __call__(self, x: list["Scalar"]):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Scalar]:
        return [weights for layer in self.layers for weights in layer.parameters()]
