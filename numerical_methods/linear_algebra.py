"""Прямые методы и вычисление характеристик матриц."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def as_float_array(values: ArrayLike) -> FloatArray:
    """Создаёт независимую копию массива с типом float64."""
    return np.array(values, dtype=np.float64, copy=True)


def gauss_solve(
    matrix: ArrayLike,
    right_hand_side: ArrayLike,
    pivoting: bool = True,
) -> tuple[FloatArray, float]:
    """Решает Ax=b методом Гаусса; возвращает решение и det(A)."""
    a = as_float_array(matrix)
    b = as_float_array(right_hand_side)

    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Матрица коэффициентов должна быть квадратной.")
    n = a.shape[0]
    if b.shape != (n,):
        raise ValueError("Вектор правой части должен иметь размерность n.")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Исходные данные должны быть конечными числами.")

    determinant_sign = 1.0

    for column in range(n - 1):
        if pivoting:
            pivot_row = column + int(np.argmax(np.abs(a[column:, column])))
            if pivot_row != column:
                a[[column, pivot_row]] = a[[pivot_row, column]]
                b[[column, pivot_row]] = b[[pivot_row, column]]
                determinant_sign *= -1.0

        pivot = a[column, column]
        if abs(pivot) <= np.finfo(float).eps:
            raise np.linalg.LinAlgError("Матрица вырождена или почти вырождена.")

        for row in range(column + 1, n):
            multiplier = a[row, column] / pivot
            a[row, column:] -= multiplier * a[column, column:]
            b[row] -= multiplier * b[column]

    if abs(a[-1, -1]) <= np.finfo(float).eps:
        raise np.linalg.LinAlgError("Матрица вырождена или почти вырождена.")

    solution = np.zeros(n, dtype=np.float64)
    for row in range(n - 1, -1, -1):
        known_sum = np.dot(a[row, row + 1:], solution[row + 1:])
        solution[row] = (b[row] - known_sum) / a[row, row]

    determinant = determinant_sign * float(np.prod(np.diag(a)))
    return solution, determinant


def inverse_gauss_jordan(matrix: ArrayLike) -> FloatArray:
    """Вычисляет обратную матрицу методом Гаусса—Жордана."""
    a = as_float_array(matrix)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Для обращения нужна квадратная матрица.")

    n = a.shape[0]
    augmented = np.hstack((a, np.eye(n, dtype=np.float64)))

    for column in range(n):
        pivot_row = column + int(np.argmax(np.abs(augmented[column:, column])))
        if abs(augmented[pivot_row, column]) <= np.finfo(float).eps:
            raise np.linalg.LinAlgError("Матрица вырождена и не имеет обратной.")

        if pivot_row != column:
            augmented[[column, pivot_row]] = augmented[[pivot_row, column]]

        augmented[column] /= augmented[column, column]

        for row in range(n):
            if row == column:
                continue
            factor = augmented[row, column]
            augmented[row] -= factor * augmented[column]

    return augmented[:, n:]


def vector_norms(vector: ArrayLike) -> dict[str, float]:
    """Возвращает нормы вектора 1, 2 и бесконечности."""
    x = as_float_array(vector)
    return {
        "1": float(np.linalg.norm(x, ord=1)),
        "2": float(np.linalg.norm(x, ord=2)),
        "inf": float(np.linalg.norm(x, ord=np.inf)),
    }


def matrix_norms(matrix: ArrayLike) -> dict[str, float]:
    """Возвращает индуцированные нормы 1, 2, бесконечности и норму Фробениуса."""
    a = as_float_array(matrix)
    return {
        "1": float(np.linalg.norm(a, ord=1)),
        "2": float(np.linalg.norm(a, ord=2)),
        "inf": float(np.linalg.norm(a, ord=np.inf)),
        "F": float(np.linalg.norm(a, ord="fro")),
    }


def condition_numbers(matrix: ArrayLike) -> dict[str, float]:
    """Вычисляет числа обусловленности в четырёх нормах."""
    a = as_float_array(matrix)
    return {
        "1": float(np.linalg.cond(a, p=1)),
        "2": float(np.linalg.cond(a, p=2)),
        "inf": float(np.linalg.cond(a, p=np.inf)),
        "F": float(np.linalg.norm(a, ord="fro") * np.linalg.norm(
            inverse_gauss_jordan(a), ord="fro"
        )),
    }


def gaussian_float32(matrix: ArrayLike, right_hand_side: ArrayLike) -> FloatArray:
    """Решает систему методом Гаусса в одинарной точности float32."""
    a = np.array(matrix, dtype=np.float32, copy=True)
    b = np.array(right_hand_side, dtype=np.float32, copy=True)
    n = a.shape[0]

    for column in range(n - 1):
        pivot_row = column + int(np.argmax(np.abs(a[column:, column])))
        if pivot_row != column:
            a[[column, pivot_row]] = a[[pivot_row, column]]
            b[[column, pivot_row]] = b[[pivot_row, column]]

        if abs(float(a[column, column])) <= np.finfo(np.float32).eps:
            raise np.linalg.LinAlgError("Нулевой ведущий элемент в float32.")

        for row in range(column + 1, n):
            multiplier = np.float32(a[row, column] / a[column, column])
            a[row, column:] -= multiplier * a[column, column:]
            b[row] -= multiplier * b[column]

    solution = np.zeros(n, dtype=np.float32)
    for row in range(n - 1, -1, -1):
        known_sum = np.dot(a[row, row + 1:], solution[row + 1:])
        solution[row] = (b[row] - known_sum) / a[row, row]

    return solution.astype(np.float64)
