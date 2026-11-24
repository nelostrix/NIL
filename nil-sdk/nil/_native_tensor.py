"""Owned contiguous float64 CPU tensors. No optional imports or implicit views."""
import itertools
import math

MAX_ELEMENTS = 100000
MAX_RANK = 8


def checked_shape(shape):
    if not isinstance(shape, (list, tuple)) or len(shape) > MAX_RANK:
        raise ValueError('Shape must have at most eight dimensions')
    shape = tuple(shape)
    if any(type(size) is not int or size <= 0 for size in shape):
        raise ValueError('Dimensions must be positive integers; empty tensors are unsupported')
    if math.prod(shape) > MAX_ELEMENTS:
        raise ValueError('Tensor exceeds 100,000-element reference limit')
    return shape


def unpack(data, depth=0):
    if type(data) in (int, float):
        if not math.isfinite(data):
            raise ValueError('Tensor values must be finite numbers')
        return (), (float(data),)
    if not isinstance(data, (list, tuple)) or not data or depth >= MAX_RANK:
        raise ValueError('Use nonempty rectangular built-in arrays, rank <=8')
    shape = None
    flat = []
    for child in data:
        child_shape, values = unpack(child, depth + 1)
        if shape is None:
            shape = child_shape
        if child_shape != shape:
            raise ValueError('Ragged tensor input')
        flat.extend(values)
        if len(flat) > MAX_ELEMENTS:
            raise ValueError('Tensor exceeds 100,000-element reference limit')
    return (len(data),) + shape, tuple(flat)


def nest(flat, shape):
    if not shape:
        return flat[0]
    if len(shape) == 1:
        return tuple(flat)
    width = math.prod(shape[1:])
    return tuple(nest(flat[start:start + width], shape[1:]) for start in range(0, len(flat), width))


def strides(shape):
    return tuple(math.prod(shape[index + 1:]) for index in range(len(shape)))


def coordinates(shape):
    return itertools.product(*(range(size) for size in shape))


def broadcast_shape(left, right):
    width = max(len(left), len(right))
    a = (1,) * (width - len(left)) + left
    b = (1,) * (width - len(right)) + right
    if any(x != y and x != 1 and y != 1 for x, y in zip(a, b)):
        raise ValueError('Incompatible broadcast shape')
    return checked_shape(tuple(max(x, y) for x, y in zip(a, b)))


def broadcast_map(shape, target):
    offset = len(target) - len(shape)
    steps = strides(shape)
    return [sum((0 if size == 1 else coordinate[offset + i]) * steps[i]
                for i, size in enumerate(shape)) for coordinate in coordinates(target)]


def same_shape_values(data, shape):
    inferred, flat = unpack(data)
    if len(shape) == 2:
        if not inferred:
            inferred = (1, 1)
        elif len(inferred) == 1:
            inferred = (1,) + inferred
    if inferred != shape:
        raise ValueError('Assigned or gradient shape differs')
    return flat


class Tensor:
    def __init__(self, data, *, shape=None, requires_grad=False, dtype='float64', device='cpu'):
        if dtype != 'float64' or device != 'cpu':
            raise ValueError('Native reference tensors support float64 CPU only')
        if type(requires_grad) is not bool:
            raise TypeError('requires_grad must be boolean')
        inferred, flat = unpack(data)
        if shape is None:
            shape = (1, 1) if not inferred else (1,) + inferred if len(inferred) == 1 else inferred
        shape = checked_shape(shape)
        if math.prod(shape) != len(flat):
            raise ValueError('Shape does not match element count')
        self._shape = shape
        self._flat = flat
        self._data = nest(flat, shape)
        self.dtype, self.device = 'float64', 'cpu'
        self.requires_grad, self.grad = requires_grad, None
        self._parents, self._versions, self._vjp, self._version = (), (), None, 0

    @property
    def data(self): return self._data
    @property
    def shape(self): return self._shape
    @property
    def ndim(self): return len(self.shape)
    @property
    def strides(self): return strides(self.shape)
    def numel(self): return len(self._flat)

    def tolist(self):
        def lists(value):
            return [lists(child) for child in value] if isinstance(value, tuple) else value
        return lists(self.data)

    def item(self):
        if self.numel() != 1:
            raise ValueError('item requires one element')
        return self._flat[0]

    def assign(self, data):
        if self._parents:
            raise RuntimeError('Assign requires a leaf tensor; intermediate graph values are immutable')
        flat = same_shape_values(data, self.shape)
        self._flat, self._data = flat, nest(flat, self.shape)
        self._version += 1

    def zero_grad(self): self.grad = None

    @classmethod
    def _op_flat(cls, values, shape, parents, vjp):
        result = cls(values, shape=shape, requires_grad=any(p.requires_grad for p in parents))
        if result.requires_grad:
            result._parents = tuple(parents)
            result._versions = tuple(p._version for p in parents)
            result._vjp = vjp
        return result

    @classmethod
    def _op(cls, data, parents, vjp):
        # Compatibility for existing public native losses and extension code.
        result = cls(data, requires_grad=any(p.requires_grad for p in parents))
        if result.requires_grad:
            result._parents = tuple(parents)
            result._versions = tuple(p._version for p in parents)
            result._vjp = lambda gradient: [(p, same_shape_values(g, p.shape))
                                           for p, g in vjp(nest(gradient, result.shape))]
        return result

    def _binary(self, other, function, derivatives):
        if not isinstance(other, Tensor):
            shape, values = unpack(other)
            other = Tensor(values, shape=shape)
        shape = broadcast_shape(self.shape, other.shape)
        left, right = broadcast_map(self.shape, shape), broadcast_map(other.shape, shape)
        values = [function(self._flat[a], other._flat[b]) for a, b in zip(left, right)]
        a_values, b_values = self._flat, other._flat
        def backward(g):
            ga, gb = [0.] * self.numel(), [0.] * other.numel()
            for value, a, b in zip(g, left, right):
                da, db = derivatives(a_values[a], b_values[b])
                ga[a] += value * da
                gb[b] += value * db
            return [(self, ga), (other, gb)]
        return Tensor._op_flat(values, shape, (self, other), backward)

    def __add__(self, other): return self._binary(other, lambda a, b: a+b, lambda a, b: (1, 1))
    __radd__ = __add__
    def __neg__(self): return self * -1
    def __sub__(self, other): return self._binary(other, lambda a, b: a-b, lambda a, b: (1, -1))
    def __rsub__(self, other): return -self + other
    def __mul__(self, other): return self._binary(other, lambda a, b: a*b, lambda a, b: (b, a))
    __rmul__ = __mul__
    def __truediv__(self, other): return self._binary(other, lambda a, b: a/b, lambda a, b: (1/b, -a/(b*b)))
    def __rtruediv__(self, other):
        shape, values = unpack(other)
        return Tensor(values, shape=shape) / self

    def __pow__(self, power):
        if type(power) not in (int, float) or not math.isfinite(power):
            raise ValueError('Power must be a finite scalar')
        return self._unary(lambda x: x**power, lambda x: 0 if power == 0 else power*x**(power-1))

    def _unary(self, function, derivative):
        data = self._flat
        return Tensor._op_flat([function(x) for x in data], self.shape, (self,),
                               lambda g: [(self, [v*derivative(x) for v, x in zip(g, data)])])

    def relu(self): return self._unary(lambda x: max(0, x), lambda x: float(x > 0))
    def tanh(self): return self._unary(math.tanh, lambda x: 1-math.tanh(x)**2)
    def sigmoid(self):
        def sigmoid(x): return 1/(1+math.exp(-x)) if x >= 0 else math.exp(x)/(1+math.exp(x))
        return self._unary(sigmoid, lambda x: sigmoid(x)*(1-sigmoid(x)))
    def exp(self): return self._unary(math.exp, math.exp)
    def log(self): return self._unary(math.log, lambda x: 1/x)

    def _axes(self, axis):
        axes = tuple(range(self.ndim)) if axis is None else (axis,) if type(axis) is int else axis
        if not isinstance(axes, (tuple, list)) or any(type(a) is not int or not -self.ndim <= a < self.ndim for a in axes):
            raise ValueError('Invalid reduction axis')
        normalized = tuple(a % self.ndim for a in axes)
        if len(set(normalized)) != len(normalized): raise ValueError('Duplicate reduction axes')
        return normalized

    def _reduction(self, axes, keepdims):
        shape = tuple(1 if i in axes else size for i, size in enumerate(self.shape)) if keepdims else tuple(size for i, size in enumerate(self.shape) if i not in axes)
        steps = strides(shape)
        mapping = []
        for coordinate in coordinates(self.shape):
            reduced = tuple(0 if i in axes else index for i, index in enumerate(coordinate)) if keepdims else tuple(index for i, index in enumerate(coordinate) if i not in axes)
            mapping.append(sum(a*b for a, b in zip(reduced, steps)))
        return shape, mapping

    def sum(self, axis=None, keepdims=False):
        if type(keepdims) is not bool: raise TypeError('keepdims must be boolean')
        shape, mapping = self._reduction(self._axes(axis), keepdims)
        if axis is None and self.ndim == 2 and not keepdims: shape = (1, 1)
        values = [0.] * math.prod(shape)
        for index, value in zip(mapping, self._flat): values[index] += value
        return Tensor._op_flat(values, shape, (self,), lambda g: [(self, [g[i] for i in mapping])])

    def mean(self, axis=None, keepdims=False):
        axes = self._axes(axis)
        return self.sum(axis=axis, keepdims=keepdims) / math.prod(self.shape[a] for a in axes)

    def log_softmax(self, axis=-1):
        axes = self._axes(axis)
        if len(axes) != 1: raise ValueError('Softmax requires one axis')
        shape, mapping = self._reduction(axes, False)
        groups = [[] for _ in range(math.prod(shape))]
        for i, group in enumerate(mapping): groups[group].append(i)
        values = [0.] * self.numel()
        for group in groups:
            peak = max(self._flat[i] for i in group)
            normalizer = math.log(sum(math.exp(self._flat[i]-peak) for i in group))
            for i in group: values[i] = self._flat[i]-peak-normalizer
        probabilities = [math.exp(value) for value in values]
        def backward(g):
            totals = [sum(g[i] for i in group) for group in groups]
            return [(self, [value-probabilities[i]*totals[mapping[i]] for i, value in enumerate(g)])]
        return Tensor._op_flat(values, self.shape, (self,), backward)

    def softmax(self, axis=-1): return self.log_softmax(axis=axis).exp()

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)): shape = tuple(shape[0])
        if any(type(size) is not int for size in shape) or shape.count(-1) > 1:
            raise ValueError('Invalid reshape dimensions')
        if -1 in shape:
            known = math.prod(size for size in shape if size != -1)
            if known <= 0 or self.numel() % known: raise ValueError('Cannot infer reshape dimension')
            shape = tuple(self.numel()//known if size == -1 else size for size in shape)
        shape = checked_shape(shape)
        if math.prod(shape) != self.numel(): raise ValueError('Reshape element count differs')
        return Tensor._op_flat(self._flat, shape, (self,), lambda g: [(self, g)])

    def permute(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)): axes = tuple(axes[0])
        if len(axes) != self.ndim or any(type(a) is not int or not -self.ndim <= a < self.ndim for a in axes):
            raise ValueError('Provide a complete axis permutation')
        axes = tuple(a % self.ndim for a in axes)
        if len(set(axes)) != self.ndim: raise ValueError('Duplicate permutation axes')
        shape = tuple(self.shape[a] for a in axes)
        source_steps = self.strides
        mapping = [sum(index*source_steps[a] for a, index in zip(axes, coordinate)) for coordinate in coordinates(shape)]
        def backward(g):
            values = [0.] * self.numel()
            for index, value in zip(mapping, g): values[index] += value
            return [(self, values)]
        return Tensor._op_flat([self._flat[i] for i in mapping], shape, (self,), backward)

    def transpose(self, first, second):
        if any(type(a) is not int or not -self.ndim <= a < self.ndim for a in (first, second)):
            raise ValueError('Invalid transpose axes')
        axes = list(range(self.ndim));first %= self.ndim;second %= self.ndim
        axes[first], axes[second] = axes[second], axes[first]
        return self.permute(axes)

    @property
    def T(self): return self.permute(tuple(reversed(range(self.ndim))))

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        if sum(item is Ellipsis for item in key) > 1: raise IndexError('Only one ellipsis is supported')
        if any(item is Ellipsis for item in key):
            at = next(i for i, item in enumerate(key) if item is Ellipsis)
            key = key[:at] + (slice(None),) * (self.ndim-len(key)+1) + key[at+1:]
        if len(key) > self.ndim: raise IndexError('Too many tensor indices')
        key += (slice(None),) * (self.ndim-len(key))
        selections, shape = [], []
        for item, size in zip(key, self.shape):
            if type(item) is int:
                index = item+size if item < 0 else item
                if not 0 <= index < size: raise IndexError('Tensor index out of bounds')
                selections.append((index,))
            elif isinstance(item, slice):
                selected = range(*item.indices(size))
                if not selected: raise ValueError('Empty tensors are unsupported')
                selections.append(selected);shape.append(len(selected))
            else: raise TypeError('Use integer/slice indices; advanced indexing is unsupported')
        mapping = [sum(a*b for a, b in zip(coordinate, self.strides)) for coordinate in itertools.product(*selections)]
        def backward(g):
            values = [0.] * self.numel()
            for index, value in zip(mapping, g): values[index] += value
            return [(self, values)]
        return Tensor._op_flat([self._flat[i] for i in mapping], tuple(shape), (self,), backward)

    def matmul(self, other, *, kernel=None):
        if not isinstance(other, Tensor) or not self.ndim or not other.ndim:
            raise ValueError('Matmul requires tensors of rank >=1')
        a_shape = (1,) + self.shape if self.ndim == 1 else self.shape
        b_shape = other.shape + (1,) if other.ndim == 1 else other.shape
        m, k = a_shape[-2:];bk, n = b_shape[-2:]
        if k != bk: raise ValueError('Matrix multiplication dimensions differ')
        batch = broadcast_shape(a_shape[:-2], b_shape[:-2])
        if math.prod(batch)*m*n > MAX_ELEMENTS:
            raise ValueError('Matrix multiplication output exceeds 100,000-element reference limit')
        shape = batch + (() if self.ndim == 1 else (m,)) + (() if other.ndim == 1 else (n,))
        a_batches, b_batches = broadcast_map(a_shape[:-2], batch), broadcast_map(b_shape[:-2], batch)
        a, b = self._flat, other._flat
        values = []
        def multiply(left, right, rows, inner, columns):
            if kernel is not None: return kernel.matmul(left, right, rows, inner, columns)
            return [sum(left[i*inner+p]*right[p*columns+j] for p in range(inner))
                    for i in range(rows) for j in range(columns)]
        for ai, bi in zip(a_batches, b_batches):
            values.extend(multiply(a[ai*m*k:(ai+1)*m*k], b[bi*k*n:(bi+1)*k*n], m, k, n))
        def backward(g):
            ga, gb = [0.] * self.numel(), [0.] * other.numel()
            for batch_index, (ai, bi) in enumerate(zip(a_batches, b_batches)):
                left, right = a[ai*m*k:(ai+1)*m*k], b[bi*k*n:(bi+1)*k*n]
                gradient = g[batch_index*m*n:(batch_index+1)*m*n]
                at = [left[i*k+p] for p in range(k) for i in range(m)]
                bt = [right[p*n+j] for j in range(n) for p in range(k)]
                da, db = multiply(gradient, bt, m, n, k), multiply(at, gradient, k, m, n)
                for index, value in enumerate(da): ga[ai*m*k+index] += value
                for index, value in enumerate(db): gb[bi*k*n+index] += value
            return [(self, ga), (other, gb)]
        return Tensor._op_flat(values, shape, (self, other), backward)

    __matmul__ = matmul

    def backward(self):
        if self.numel() != 1 or not self.requires_grad: raise ValueError('backward requires a differentiable scalar')
        order, seen, stack = [], set(), [(self, False)]
        while stack:
            node, visited = stack.pop()
            if visited: order.append(node);continue
            if node in seen: continue
            seen.add(node);stack.append((node, True));stack.extend((p, False) for p in reversed(node._parents))
        for node in order:
            if tuple(p._version for p in node._parents) != node._versions:
                raise RuntimeError('A tensor changed after this graph was constructed')
        gradients = {self: [1.]}
        for node in reversed(order):
            if node not in gradients or node._vjp is None: continue
            for parent, values in node._vjp(gradients[node]):
                if not parent.requires_grad: continue
                if len(values) != parent.numel(): raise ValueError('Invalid backward shape')
                current = gradients.setdefault(parent, [0.] * parent.numel())
                for i, value in enumerate(values): current[i] += value
        updates = []
        for node, values in gradients.items():
            if node._parents or not node.requires_grad: continue
            previous = same_shape_values(node.grad, node.shape) if node.grad is not None else [0.] * node.numel()
            combined = [a+b for a, b in zip(values, previous)]
            # Validate every accumulated leaf before committing any gradients.
            _, checked = unpack(combined)
            updates.append((node, nest(checked, node.shape)))
        for node, values in updates: node.grad = values
