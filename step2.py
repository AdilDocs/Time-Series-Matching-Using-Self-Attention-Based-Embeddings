"""
Step 2: Minimal Reverse-Mode Automatic Differentiation Engine
==================================================================
A compact autodiff engine for real-valued numpy arrays, supporting
broadcasting. Provides exactly the operations the self-attention encoder
(step3.py) needs: elementwise arithmetic, matrix multiplication, softmax,
layer normalization, ReLU, and reduction/reshape operations -- each with
an exact backward pass.
"""

import numpy as np


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Var:
    """A node in the computation graph, wrapping a real-valued numpy array."""

    def __init__(self, value, _prev=()):
        self.value = np.asarray(value, dtype=np.float64)
        self.shape = self.value.shape
        self.grad = np.zeros_like(self.value)
        self._backward = lambda: None
        self._prev = _prev

    def __add__(self, other):
        other = other if isinstance(other, Var) else Var(np.asarray(other, dtype=np.float64))
        out = Var(self.value + other.value, (self, other))

        def _backward():
            self.grad += _unbroadcast(out.grad, self.shape)
            other.grad += _unbroadcast(out.grad, other.shape)

        out._backward = _backward
        return out

    __radd__ = __add__

    def __neg__(self):
        out = Var(-self.value, (self,))

        def _backward():
            self.grad += -out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Var) else Var(np.asarray(other, dtype=np.float64))
        return self + (-other)

    def __rsub__(self, other):
        other = other if isinstance(other, Var) else Var(np.asarray(other, dtype=np.float64))
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Var) else Var(np.asarray(other, dtype=np.float64))
        out = Var(self.value * other.value, (self, other))

        def _backward():
            self.grad += _unbroadcast(out.grad * other.value, self.shape)
            other.grad += _unbroadcast(out.grad * self.value, other.shape)

        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __pow__(self, power):
        out = Var(self.value ** power, (self,))

        def _backward():
            self.grad += out.grad * power * (self.value ** (power - 1))

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Var) else Var(np.asarray(other, dtype=np.float64))
        return self * (other ** -1.0)

    def sum(self):
        out = Var(np.array(self.value.sum()), (self,))

        def _backward():
            self.grad += np.ones_like(self.value) * out.grad

        out._backward = _backward
        return out

    def mean(self):
        n = self.value.size
        return self.sum() * (1.0 / n)

    def reshape(self, *shape):
        old_shape = self.shape
        out = Var(self.value.reshape(*shape), (self,))

        def _backward():
            self.grad += out.grad.reshape(old_shape)

        out._backward = _backward
        return out

    def take_indices(self, indices: np.ndarray):
        out = Var(self.value[indices], (self,))

        def _backward():
            np.add.at(self.grad, indices, out.grad)

        out._backward = _backward
        return out

    def roll(self, shift: int, axis: int = -1):
        out = Var(np.roll(self.value, shift, axis=axis), (self,))

        def _backward():
            self.grad += np.roll(out.grad, -shift, axis=axis)

        out._backward = _backward
        return out

    def __getitem__(self, idx):
        out = Var(self.value[idx], (self,))

        def _backward():
            self.grad[idx] += out.grad

        out._backward = _backward
        return out

    def sin(self):
        out = Var(np.sin(self.value), (self,))

        def _backward():
            self.grad += out.grad * np.cos(self.value)

        out._backward = _backward
        return out

    def cos(self):
        out = Var(np.cos(self.value), (self,))

        def _backward():
            self.grad += out.grad * (-np.sin(self.value))

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.value))
        out = Var(s, (self,))

        def _backward():
            self.grad += out.grad * s * (1.0 - s)

        out._backward = _backward
        return out

    def relu(self):
        out = Var(np.maximum(self.value, 0.0), (self,))

        def _backward():
            self.grad += out.grad * (self.value > 0.0)

        out._backward = _backward
        return out

    def matmul(self, other):
        other = other if isinstance(other, Var) else Var(np.asarray(other, dtype=np.float64))
        out = Var(self.value @ other.value, (self, other))

        def _backward():
            a, b = self.value, other.value
            grad_out = out.grad
            self.grad += _unbroadcast(grad_out @ np.swapaxes(b, -1, -2), self.shape)
            other.grad += _unbroadcast(np.swapaxes(a, -1, -2) @ grad_out, other.shape)

        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        x = self.value
        x_shift = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shift)
        s = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        out = Var(s, (self,))

        def _backward():
            g = out.grad
            dot = np.sum(g * s, axis=axis, keepdims=True)
            self.grad += s * (g - dot)

        out._backward = _backward
        return out

    def transpose_last2(self):
        out = Var(np.swapaxes(self.value, -1, -2), (self,))

        def _backward():
            self.grad += np.swapaxes(out.grad, -1, -2)

        out._backward = _backward
        return out

    def layernorm(self, axis=-1, eps=1e-5):
        x = self.value
        mu = np.mean(x, axis=axis, keepdims=True)
        var = np.var(x, axis=axis, keepdims=True)
        std = np.sqrt(var + eps)
        x_norm = (x - mu) / std
        out = Var(x_norm, (self,))
        n = x.shape[axis]

        def _backward():
            g = out.grad
            dxnorm = g
            dvar_term = np.sum(dxnorm * (x - mu), axis=axis, keepdims=True) * (-0.5) * (var + eps) ** (-1.5)
            dmu_term = np.sum(dxnorm * (-1.0 / std), axis=axis, keepdims=True) + \
                       dvar_term * np.mean(-2.0 * (x - mu), axis=axis, keepdims=True)
            self.grad += dxnorm / std + dvar_term * 2.0 * (x - mu) / n + dmu_term / n

        out._backward = _backward
        return out

    def maximum_scalar(self, c: float):
        out = Var(np.maximum(self.value, c), (self,))

        def _backward():
            self.grad += out.grad * (self.value > c)

        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()

        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for p in v._prev:
                    build(p)
                topo.append(v)

        build(self)
        self.grad = np.ones_like(self.value)
        for v in reversed(topo):
            v._backward()


def concat_last_axis(vars_list):
    """Concatenates a list of Vars along the last axis. Used to combine
    multi-head attention outputs."""
    values = [v.value for v in vars_list]
    out = Var(np.concatenate(values, axis=-1), tuple(vars_list))
    sizes = [v.shape[-1] for v in vars_list]
    offsets = np.cumsum([0] + sizes)

    def _backward():
        for i, v in enumerate(vars_list):
            v.grad += out.grad[..., offsets[i]:offsets[i + 1]]

    out._backward = _backward
    return out


if __name__ == "__main__":
    print("=" * 60)
    print("Autodiff engine self-check")
    print("=" * 60)
    np.random.seed(0)
    eps = 1e-6

    A_val = np.random.randn(3, 4)
    B_val = np.random.randn(4, 2)
    A = Var(A_val.copy())
    B = Var(B_val.copy())
    C = A.matmul(B)
    loss = (C * C).sum()
    loss.backward()
    num_grad_A = np.zeros_like(A_val)
    for i in range(3):
        for j in range(4):
            Ap = A_val.copy(); Ap[i, j] += eps
            Am = A_val.copy(); Am[i, j] -= eps
            num_grad_A[i, j] = (np.sum((Ap @ B_val) ** 2) - np.sum((Am @ B_val) ** 2)) / (2 * eps)
    print(f"matmul gradient max diff: {np.max(np.abs(A.grad - num_grad_A)):.2e}")

    x_val = np.random.randn(3, 5)
    x = Var(x_val.copy())
    s = x.softmax(axis=-1)
    target = Var(np.random.randn(3, 5))
    loss2 = (s * target).sum()
    loss2.backward()

    def sm(v):
        e = np.exp(v - v.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    num_grad_x = np.zeros_like(x_val)
    for i in range(3):
        for j in range(5):
            xp = x_val.copy(); xp[i, j] += eps
            xm = x_val.copy(); xm[i, j] -= eps
            num_grad_x[i, j] = (np.sum(sm(xp) * target.value) - np.sum(sm(xm) * target.value)) / (2 * eps)
    print(f"softmax gradient max diff: {np.max(np.abs(x.grad - num_grad_x)):.2e}")

    x2_val = np.random.randn(3, 6) * 3 + 5
    x2 = Var(x2_val.copy())
    ln = x2.layernorm(axis=-1)
    target2 = Var(np.random.randn(3, 6))
    loss3 = (ln * target2).sum()
    loss3.backward()

    def ln_np(v):
        mu = v.mean(axis=-1, keepdims=True)
        var = v.var(axis=-1, keepdims=True)
        return (v - mu) / np.sqrt(var + 1e-5)

    num_grad_x2 = np.zeros_like(x2_val)
    for i in range(3):
        for j in range(6):
            xp = x2_val.copy(); xp[i, j] += eps
            xm = x2_val.copy(); xm[i, j] -= eps
            num_grad_x2[i, j] = (np.sum(ln_np(xp) * target2.value) - np.sum(ln_np(xm) * target2.value)) / (2 * eps)
    print(f"layernorm gradient max diff: {np.max(np.abs(x2.grad - num_grad_x2)):.2e}")
    print("\nAll diffs should be near 0 (~1e-8 to 1e-10) if correct.")
