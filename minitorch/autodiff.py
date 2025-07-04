from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    if arg == 0:
        return (f(vals[0] + epsilon, *vals[1:]) - f(vals[0] - epsilon, *vals[1:])) / (
            2 * epsilon
        )

    if arg == len(vals) - 1:
        return (
            f(*vals[:-1], vals[-1] + epsilon) - f(*vals[:-1], vals[-1] - epsilon)
        ) / (2 * epsilon)

    return (
        f(*vals[:arg], vals[arg] + epsilon, *vals[arg + 1 :])
        - f(*vals[:arg], vals[arg] - epsilon, *vals[arg + 1 :])
    ) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    topo = []
    # visited = set()

    def _topological_sort(var: Variable) -> None:
        # if var.unique_id in visited:
        #     return
        # visited.add(var.unique_id)
        if not var.is_constant():
            for parent in var.parents:
                _topological_sort(parent)
        topo.append(var)

    _topological_sort(variable)
    return topo[::-1]


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # Correct implementation for Task 1.4.
    derivatives = {variable.unique_id: deriv}
    for var in topological_sort(variable):
        d_output = derivatives.get(var.unique_id, 0.0)
        if var.is_leaf():
            var.accumulate_derivative(d_output)
        elif var.is_constant():
            continue
        else:
            for parent, d_parent in var.chain_rule(d_output):
                derivatives[parent.unique_id] = d_parent


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
