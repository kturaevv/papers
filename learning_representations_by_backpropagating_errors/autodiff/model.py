import math


class Scalar:
    def __init__(self, data, prev=()) -> None:
        self.id = id(self)
        self.data: float = data
        self.grad: float = 0
        self.prev: tuple["Scalar"] = prev
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Scalar(data={self.data})"

    def __add__(self, other: "Scalar") -> "Scalar":
        if not isinstance(other, Scalar):
            other = Scalar(other)

        def _backward(prev: tuple["Scalar", "Scalar"], d_out) -> tuple[float]:
            return d_out, d_out

        out = Scalar(self.data + other.data, prev=(self, other))
        out._backward = _backward

        return out

    def __mul__(self, other: "Scalar") -> "Scalar":
        if not isinstance(other, Scalar):
            other = Scalar(other)

        def _backward(prev: tuple["Scalar", "Scalar"], d_out) -> tuple[float]:
            lval, rval = prev
            return rval.data * d_out, lval.data * d_out

        out = Scalar(self.data * other.data, prev=(self, other))
        out._backward = _backward

        return out

    def __pow__(self, other: int):
        if isinstance(other, Scalar):
            other = other.data

        def _backward(prev, d_out):
            val, other = prev
            return (other * (val.data ** (other - 1)) * d_out,)

        out = Scalar(self.data**other, prev=(self, other))
        out._backward = _backward
        return out

    def sigmoid(self) -> "Scalar":
        sigmoid_fn = lambda x: 1 / (1 + math.exp(-x))  # noqa: E731

        def _backward(prev: tuple["Scalar"], d_out) -> tuple[float]:
            d_sigmoid = (
                sigmoid_fn(prev[0].data) * (1 - sigmoid_fn(prev[0].data)) * d_out
            )
            return (d_sigmoid,)

        out = Scalar(sigmoid_fn(self.data), (self,))
        out._backward = _backward
        return out

    def tanh(self) -> "Scalar":
        def _backward(prev: tuple["Scalar"], d_out) -> tuple[float]:
            d_tanh = (1 - math.tanh(prev[0].data) ** 2) * d_out
            return (d_tanh,)

        out = Scalar(math.tanh(self.data), (self,))
        out._backward = _backward

        return out

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

    """ Backpropagation """

    def in_chain(self):
        return len(self.prev) > 0

    def _backward():
        """
        Function attached to every arithmetic operation to calculate the derivative in the chain

        Backward function keeps reference to [self], [other] and [out] Scalar value, and is directly tied to the arithmetic primitive.
        """
        ...

    @staticmethod
    def topological_sort(v) -> list["Scalar"]:
        visited = set()
        topo = [v]
        qu = [v]
        while qu:
            node = qu.pop(0)

            if node.id in visited:
                continue
            visited.add(node.id)

            for child in node.prev:
                if not isinstance(child, Scalar) or child.in_chain() is False:
                    continue
                qu.append(child)
                topo.append(child)
        return topo

    def backward(self):
        node_grads = {}
        node_grads[self.id] = 1

        topo = self.topological_sort(self)

        for node in topo:
            deriv = node_grads[node.id]

            for scalar, grad in zip(node.prev, node._backward(node.prev, deriv)):
                scalar.grad += grad
                if scalar not in node_grads:
                    node_grads[scalar.id] = scalar.grad
                else:
                    node_grads[scalar.id] += scalar.grad

    @classmethod
    def check_autograd(cls):
        "Compare this implementation to Torch"

        from torch import Tensor  # type: ignore

        a = Tensor([3])
        b = Tensor([7])
        c = Tensor([9])
        d = Tensor([1])

        a.requires_grad = True
        b.requires_grad = True
        c.requires_grad = True
        d.requires_grad = True

        d = a.tanh() ** 1.02 + b.tanh() * c.tanh() / 3 * d.tanh()
        d.backward()

        print(a, a.grad)
        print(b, b.grad)

        a = Scalar(3)
        b = Scalar(7)
        c = Scalar(9)
        d = Scalar(1)

        e = a.tanh() ** 1.02 + b.tanh() * c.tanh() / 3 * d.tanh()
        e.backward()

        print(a)
        print(b)
