"""Итерационные методы Якоби (простых итераций) и Зейделя."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass
class IterationResult:
    """Результат итерационного процесса и история его изменения."""

    solution: FloatArray
    iterations: int
    status: str
    step_history: list[float] = field(default_factory=list)
    error_history: list[float] = field(default_factory=list)
    residual_history: list[float] = field(default_factory=list)


def to_iterative_form(
    matrix: ArrayLike,
    right_hand_side: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Преобразует Ax=b к виду x=B x+c для метода простых итераций."""
    a = np.asarray(matrix, dtype=np.float64)
    b = np.asarray(right_hand_side, dtype=np.float64)

    if a.ndim != 2 or a.shape[0] != a.shape[1] or b.shape != (a.shape[0],):
        raise ValueError("Некорректные размеры матрицы или вектора.")
    if np.any(np.abs(np.diag(a)) <= np.finfo(float).eps):
        raise ZeroDivisionError("На диагонали есть нулевой элемент.")

    diagonal = np.diag(a)
    b_matrix = -a / diagonal[:, None]
    np.fill_diagonal(b_matrix, 0.0)
    c_vector = b / diagonal
    return b_matrix, c_vector


def spectral_radius(matrix: ArrayLike) -> float:
    """Спектральный радиус матрицы — максимум модулей её собственных значений."""
    eigenvalues = np.linalg.eigvals(np.asarray(matrix, dtype=np.float64))
    return float(np.max(np.abs(eigenvalues)))


def seidel_iteration_matrix(b_matrix: ArrayLike) -> FloatArray:
    """Строит матрицу итераций Зейделя T=(I-L)^(-1)U."""
    b = np.asarray(b_matrix, dtype=np.float64)
    lower = np.tril(b, k=-1)
    upper = np.triu(b, k=1)
    return np.linalg.solve(np.eye(b.shape[0]) - lower, upper)


def _run_iterations(
    b_matrix: FloatArray,
    c_vector: FloatArray,
    method: str,
    epsilon: float,
    max_iterations: int,
    exact_solution: ArrayLike | None,
    original_matrix: FloatArray,
    original_rhs: FloatArray,
    initial_approximation: ArrayLike | None,
) -> IterationResult:
    n = c_vector.size
    x = (
        np.zeros(n, dtype=np.float64)
        if initial_approximation is None
        else np.array(initial_approximation, dtype=np.float64, copy=True)
    )
    if x.shape != (n,):
        raise ValueError("Начальное приближение имеет неверную размерность.")

    exact = None if exact_solution is None else np.asarray(exact_solution, dtype=np.float64)
    step_history: list[float] = []
    error_history: list[float] = []
    residual_history: list[float] = []
    residual_tolerance = epsilon * max(1.0, float(np.linalg.norm(original_rhs, ord=np.inf)))
    previous_step = float("inf")
    stagnant_iterations = 0

    for iteration in range(1, max_iterations + 1):
        old_x = x.copy()

        if method == "jacobi":
            x = b_matrix @ old_x + c_vector
        else:
            for i in range(n):
                x[i] = float(np.dot(b_matrix[i], x) + c_vector[i])

        step_norm = float(np.linalg.norm(x - old_x, ord=np.inf))
        residual_norm = float(np.linalg.norm(original_rhs - original_matrix @ x, ord=np.inf))

        step_history.append(step_norm)
        residual_history.append(residual_norm)

        if exact is not None:
            error_history.append(float(np.linalg.norm(x - exact, ord=np.inf)))

        if not np.all(np.isfinite(x)) or not np.isfinite(step_norm):
            return IterationResult(x, iteration, "расходимость: получены нечисловые значения",
                                   step_history, error_history, residual_history)

        if step_norm < epsilon and residual_norm < residual_tolerance:
            return IterationResult(x, iteration, "сошёлся", step_history,
                                   error_history, residual_history)

        if step_norm >= previous_step * (1.0 - 1e-13):
            stagnant_iterations += 1
        else:
            stagnant_iterations = 0

        if stagnant_iterations >= 200:
            return IterationResult(x, iteration, "остановлен: стагнация итераций",
                                   step_history, error_history, residual_history)

        if np.linalg.norm(x, ord=np.inf) > 1e100:
            return IterationResult(x, iteration, "расходимость: рост решения",
                                   step_history, error_history, residual_history)

        previous_step = step_norm

    return IterationResult(x, max_iterations, "не достигнута точность за лимит итераций",
                           step_history, error_history, residual_history)


def solve_iterative(
    b_matrix: ArrayLike,
    c_vector: ArrayLike,
    method: str,
    original_matrix: ArrayLike,
    original_rhs: ArrayLike,
    epsilon: float = 1e-10,
    max_iterations: int = 20_000,
    exact_solution: ArrayLike | None = None,
    initial_approximation: ArrayLike | None = None,
) -> IterationResult:
    """Запускает метод Якоби или Зейделя с контролем шага и невязки."""
    if epsilon <= 0:
        raise ValueError("Точность epsilon должна быть положительной.")
    if max_iterations < 1:
        raise ValueError("Лимит итераций должен быть положительным.")
    if method not in {"jacobi", "seidel"}:
        raise ValueError("Метод должен быть 'jacobi' или 'seidel'.")

    b = np.asarray(b_matrix, dtype=np.float64)
    c = np.asarray(c_vector, dtype=np.float64)
    a_original = np.asarray(original_matrix, dtype=np.float64)
    rhs_original = np.asarray(original_rhs, dtype=np.float64)

    return _run_iterations(
        b, c, method, epsilon, max_iterations, exact_solution,
        a_original, rhs_original, initial_approximation
    )
