# Learning representations by back-propagating errors

---
**Quick summary:** *paper that introduced and popularized the concept of backpropagation for Neural Networks*

> The task is specified by giving the desired state vector of the output units for each state vector of the input units

Quite convoluted way of saying that to make a network train we need to provide it the intput data with corresponding output we hope to achieve, i.e. a supervised learning. In other words, to map inputs to the outputs with the data.

> ... state of the units in each layer are determined by applying equations (1) and (2) ...

Description of how Neural Networks work, i.e. every output of a layer should correspond to every input node of next layer. And layers are pieced together by a linking linear function with non-linearity.

A Unit consist of 2 values which is a **weight** and a **bias**. In a modern terminology this is reffered to as a *Neuron*

> The total input, $x_{j}$ to unit j is a linear function of the outputs,
> $y_{i}$ of the units that are connected to j and of the weights, $w_{ji}$
> on these connections $ x_{j} = \sum_{i}y_{i}w_{ji} $

... the states of the "units", aka Neurons, are evaluated by the linear relation of inputs, which means that an output of neuron $x_{j}$, is a sum of weights of every input to every weight of the Neuron:

```python
    def __call__(self, y: "Units") -> "Units":
        x_j = sum(y_i * w_ji for y_i, w_ji in zip(y.w, self.w))
        return x_j
```

> A unit has a real-valued output, $y_{j}$ which is a non-linear function of its total input $ y_{j} = {1 \over 1 + e^{-x_{j}} }$

This is the non-linearity that makes the network learn stuff, which is commonly reffered to as *activation function*. The one mentioned in the paper is *Sigmoid activation function*.

```python
    def sigmoid(self) -> "Scalar":
        return 1 / (1 + math.exp(-self.w))

    def __call__(self, y: "Units") -> "Units":
        x_j = sum(y_i * w_ji for y_i, w_ji in zip(y.w, self.w))
        out = x_j.sigmoid() # add non-linearity
        return out
```

Based on above mentioned ideas it is enough to "build" the basic implementation for a simplest, non-optimized neural network, featuring:
    - Scalar: the most basic building block of a Neural network
    - Units: aka Neuron
    - State Vector: aka Tensor


```python
import math
import random

class Scalar:
    
    def __init__(self, data = None) -> None:
        self.w: float = data if data is not None else random.uniform(-1, 1)

    def __repr__(self) -> str:
        return f"Scalar(w={self.w})"

    def __add__(self, other: "Scalar") -> "Scalar":
        if not isinstance(other, Scalar):
            other = Scalar(other)

        self.w += other.w
        return self

    def __mul__(self, other: "Scalar") -> "Scalar":
        if not isinstance(other, Scalar):
            other = Scalar(other)

        self.w *= other.w
        return self

    def __pow__(self, power: int):
        if isinstance(power, Scalar):
            power = power.w
        self.w ^= power
        return self
    
    """ Handle other use cases using base operations. """

    def __neg__(self) -> "Scalar":  # -self
        return self * -1

    def __radd__(self, other: "Scalar") -> "Scalar":  # other + self
        return self + other

    def __sub__(self, other: "Scalar") -> "Scalar":  # self - other
        return self + (-other)

    def __rsub__(self, other: "Scalar") -> "Scalar":  # other - self
        return other + (-self)

    def __rmul__(self, other: "Scalar") -> "Scalar":  # other * self
        return self * other

    def __truediv__(self, other: "Scalar") -> "Scalar":  # self / other
        return self * other**-1

    def __rtruediv__(self, other: "Scalar") -> "Scalar":  # other / self
        return other * self**-1

    def sigmoid(self) -> "Scalar":
        return 1 / (1 + math.exp(-self.w))

class Units: # aka a single Neuron
    
    def __init__(self, n_in: int) -> None:
        self.w: list[Scalar] = [Scalar() for _ in range(n_in)]
        self.bias: float = 1.0

    def __call__(self, y: list[float]) -> "Units":
        if len(y) != len(self.w): 
            raise BaseException("Incorrect Dimensions")

        x_j: Scalar = sum((y_i * w_ji for y_i, w_ji in zip(y, self.w)))
        out = x_j.sigmoid()
        return out

    def __repr__(self) -> str:
        doc: str = "Unit[\n"
        for x_i in self.w:
            doc += "\t" + x_i.__str__() + "\n"
        doc += "]"
        return doc

class StateVector: # aka Tensor
    
    def __init__(self, n_in: int, n_out: int) -> None:
        self.n_in: int = n_in
        self.n_out: int = n_out
        self.units: list[Units] = [Units(n_in) for _ in range(n_out)]

    def __call__(self, y: list[float]) -> "StateVector":
        out: list[Units] = [unit(y) for unit in self.units]
        return out

    def __repr__(self) -> str:
        doc: str = f"StateVector[{self.n_in}, {self.n_out}][\n"
        for j in self.units:
            doc += "  " + j.__str__() + "\n"
        doc += "]"
        return doc

class NeuralNetwork: 
    def __init__(self, n_in, layers: list[StateVector]) -> None:
        self.n_in = n_in
        self.layers: list[StateVector] = layers

    def __call__(self, x: list[float]):
        for layer in self.layers:
            x = layer(x)
        return x

mlp = NeuralNetwork(3, [
    StateVector(3, 10),
    StateVector(10, 10),
    StateVector(10, 1)
])
mlp([1,2,3])
```




    [0.30180932633148394]




