"""Эксперименты и построение графиков для практической работы № 1."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from numerical_methods.iterative_methods import (
    seidel_iteration_matrix,
    solve_iterative,
    spectral_radius,
    to_iterative_form,
)
from numerical_methods.linear_algebra import (
    condition_numbers,
    gauss_solve,
    gaussian_float32,
    inverse_gauss_jordan,
    matrix_norms,
)


A1 = np.array([
    [42.78, 16.87, -4.41, -40.83],
    [-13.53, -4.46, 4.08, 10.52],
    [-33.22, -10.60, 7.88, 26.49],
    [46.65, 15.24, -11.27, -37.43],
], dtype=float)
b1 = np.array([-128.85, 38.65, 94.08, -132.78], dtype=float)

A2 = np.array([
    [-6.18, -6.41, 4.26, -1.61],
    [-2.37, 4.01, -0.73, -8.82],
    [4.26, 1.98, 8.78, -0.97],
    [-6.17, 6.24, 2.07, 4.33],
], dtype=float)
b2 = np.array([7.22, -16.0, 2.58, 16.9], dtype=float)

EPSILON = 1e-10
MAX_ITERATIONS = 20_000
OUTPUT_DIR = Path("results")
EXACT_SOLUTION = np.array([-1.0, 0.0, 1.0, 2.0])


def print_direct_method_results(name: str, matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    solution, determinant = gauss_solve(matrix, rhs)
    inverse = inverse_gauss_jordan(matrix)
    residual = rhs - matrix @ solution
    norms_a = matrix_norms(matrix)
    norms_inverse = matrix_norms(inverse)
    conditions = condition_numbers(matrix)

    print(f"\nСистема {name}")
    print(f"Решение методом Гаусса: {solution}")
    print(f"Проверка NumPy:         {np.linalg.solve(matrix, rhs)}")
    print(f"Определитель: {determinant:.10g}")
    print(f"Норма невязки ||b-Ax||inf: {np.linalg.norm(residual, ord=np.inf):.4e}")
    print(f"||A||1={norms_a['1']:.6g}; ||A||2={norms_a['2']:.6g}; "
          f"||A||inf={norms_a['inf']:.6g}; ||A||F={norms_a['F']:.6g}")
    print(f"||A^-1||1={norms_inverse['1']:.6g}; ||A^-1||2={norms_inverse['2']:.6g}; "
          f"||A^-1||inf={norms_inverse['inf']:.6g}; ||A^-1||F={norms_inverse['F']:.6g}")
    print(f"cond1={conditions['1']:.6g}; cond2={conditions['2']:.6g}; "
          f"cond_inf={conditions['inf']:.6g}; condF={conditions['F']:.6g}")

    return solution


def build_transforms() -> list[tuple[str, np.ndarray, np.ndarray, str]]:
    """Возвращает преобразования, используемые в отчёте."""
    approximate_inverse_a1 = np.round(inverse_gauss_jordan(A1), decimals=2)
    return [
        ("A1: исходная", A1, b1, "A1"),
        ("A1: нормальная A^T A", A1.T @ A1, A1.T @ b1, "A1"),
        ("A1: предобусловленная M A", approximate_inverse_a1 @ A1,
         approximate_inverse_a1 @ b1, "A1"),
        ("A2: исходная", A2, b2, "A2"),
        ("A2: нормальная A^T A", A2.T @ A2, A2.T @ b2, "A2"),
    ]


def iteration_history(
    b_matrix: np.ndarray,
    c_vector: np.ndarray,
    method: str,
    exact: np.ndarray = EXACT_SOLUTION,
    iterations: int = 100,
) -> tuple[list[float], list[float]]:
    """Возвращает нормы шага и истинной погрешности по каждой итерации."""
    x = np.zeros_like(c_vector, dtype=float)
    steps: list[float] = []
    errors: list[float] = []
    for _ in range(iterations):
        old = x.copy()
        if method == "jacobi":
            x = b_matrix @ old + c_vector
        elif method == "seidel":
            for i in range(x.size):
                x[i] = b_matrix[i] @ x + c_vector[i]
        else:
            raise ValueError("Неизвестный метод итераций")
        steps.append(float(np.linalg.norm(x - old, ord=np.inf)))
        errors.append(float(np.linalg.norm(x - exact, ord=np.inf)))
        if not np.all(np.isfinite(x)) or np.linalg.norm(x, ord=np.inf) > 1e100:
            break
    return steps, errors


def count_iterations(b_matrix: np.ndarray, c_vector: np.ndarray,
                     method: str, epsilon: float, max_iterations: int = 100_000) -> int:
    """Число итераций до выполнения критерия ||x(k)-x(k-1)||inf < epsilon."""
    x = np.zeros_like(c_vector, dtype=float)
    for k in range(1, max_iterations + 1):
        old = x.copy()
        if method == "jacobi":
            x = b_matrix @ old + c_vector
        else:
            for i in range(x.size):
                x[i] = b_matrix[i] @ x + c_vector[i]
        if not np.all(np.isfinite(x)):
            return k
        if np.linalg.norm(x - old, ord=np.inf) < epsilon:
            return k
    return max_iterations


def _plot_perturbation(axis, solutions: dict[str, np.ndarray]) -> None:
    """Строит рисунок 1 в формате отчёта: фактическая ошибка и оценка cond∞·δb."""
    # Возмущение задаётся в процентах. 12 логарифмически распределённых
    # значений дают плавные прямые, как на рисунке 1 отчёта.
    deltas_percent = np.logspace(-3, 0.5, 12)
    signs = np.array([1.0, -1.0, 1.0, -1.0])
    cases = (
        ("Система 1", A1, b1, "black", "o"),
        ("Система 2", A2, b2, "#777777", "s"),
    )

    for name, matrix, rhs, color, marker in cases:
        exact = solutions[name]
        condition = condition_numbers(matrix)["inf"]
        actual_errors = []
        for delta_percent in deltas_percent:
            delta = delta_percent / 100.0
            changed_rhs = rhs + delta * np.abs(rhs) * signs
            changed_solution, _ = gauss_solve(matrix, changed_rhs)
            relative_error_percent = (
                np.linalg.norm(changed_solution - exact, ord=np.inf)
                / np.linalg.norm(exact, ord=np.inf) * 100.0
            )
            actual_errors.append(relative_error_percent)

        # Оценка (12): относительная ошибка решения не превосходит
        # cond∞(A) умножить на относительное возмущение правой части.
        estimate_percent = condition * deltas_percent
        axis.loglog(
            deltas_percent, actual_errors,
            color=color, marker=marker, markersize=6, linewidth=2.0,
            label=f"{name}: фактическая погрешность",
        )
        axis.loglog(
            deltas_percent, estimate_percent,
            color=color, linestyle=":", linewidth=2.2,
            label=rf"{name}: оценка $\mathrm{{cond}}_\infty \cdot \delta b$",
        )

    axis.set_xlabel(r"Относительное возмущение $||\delta b||/||b||$, %", fontsize=12)
    axis.set_ylabel(r"$||\delta x||/||x||$, %", fontsize=12)
    axis.set_xlim(8e-4, 4.8)
    axis.grid(True, which="both", color="#b0b0b0", alpha=0.35, linewidth=0.8)
    axis.legend(loc="upper left", fontsize=9, frameon=False)
    axis.tick_params(axis="both", which="major", labelsize=10)


def save_plots(solutions: dict[str, np.ndarray]) -> None:
    """Создаёт ровно шесть графиков, перечисленных в отчёте."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Удаляем файлы от предыдущей версии программы, чтобы не оставались
    # лишние графики и CSV, которые больше не формируются.
    for obsolete_name in (
        "iteration_history.csv", "convergence.png", "spectral_radii.png"
    ):
        obsolete_path = OUTPUT_DIR / obsolete_name
        if obsolete_path.exists():
            obsolete_path.unlink()

    # Рисунок 1. Относительная погрешность решения от возмущения правой части.
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    # Единая типографика и пропорции близки к рисункам в отчёте.
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix"})
    _plot_perturbation(ax, solutions)
    fig.savefig(OUTPUT_DIR / "01_relative_error_vs_rhs_perturbation.png", dpi=180)
    plt.close(fig)

    # Рисунок 2. Расходимость обоих методов для исходных систем.
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, matrix, rhs in (("Система 1", A1, b1), ("Система 2", A2, b2)):
        b_matrix, c_vector = to_iterative_form(matrix, rhs)
        for method, method_title, style in (
            ("jacobi", "простые итерации", "-"),
            ("seidel", "Зейдель", "--"),
        ):
            _, errors = iteration_history(b_matrix, c_vector, method, iterations=1200)
            if errors:
                ax.semilogy(np.arange(1, len(errors) + 1), errors,
                            linestyle=style, label=f"{name}: {method_title}")
    ax.set_xlabel("Номер итерации")
    ax.set_ylabel(r"Погрешность $||x^{(k)}-x^*||_\infty$")
    ax.set_title("Расходимость итерационных методов для исходных систем")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_divergence_original_systems.png", dpi=180)
    plt.close(fig)

    # Рисунки 3 и 4. Погрешность и разность приближений для успешных преобразований.
    convergence_cases = [
        ("Система 1 (после предобусловливания)",
         np.round(inverse_gauss_jordan(A1), 2) @ A1,
         np.round(inverse_gauss_jordan(A1), 2) @ b1,
         "03_convergence_preconditioned_system1.png"),
        ("Система 2 (нормальная система)", A2.T @ A2, A2.T @ b2,
         "04_convergence_normal_system2.png"),
    ]
    for title, matrix, rhs, filename in convergence_cases:
        b_matrix, c_vector = to_iterative_form(matrix, rhs)
        fig, ax = plt.subplots(figsize=(9, 6))
        for method, method_title, style in (
            ("jacobi", "Простые итерации", "-"),
            ("seidel", "Метод Зейделя", "--"),
        ):
            steps, errors = iteration_history(b_matrix, c_vector, method, iterations=80)
            k = np.arange(1, len(steps) + 1)
            ax.semilogy(k, errors, linestyle=style, marker="o", markersize=3,
                        label=f"{method_title}: погрешность")
            ax.semilogy(k, steps, linestyle=style, marker="x", markersize=3,
                        label=f"{method_title}: разность приближений")
        ax.set_xlabel("Номер итерации")
        ax.set_ylabel("Норма вектора (логарифмическая шкала)")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=180)
        plt.close(fig)

    # Рисунок 5. Ложная сходимость метода Зейделя для нормальной системы 1.
    normal_a1 = A1.T @ A1
    normal_b1 = A1.T @ b1
    b_matrix, c_vector = to_iterative_form(normal_a1, normal_b1)
    steps, errors = iteration_history(b_matrix, c_vector, "seidel", iterations=20_000)
    fig, ax = plt.subplots(figsize=(9, 6))
    k = np.arange(1, len(steps) + 1)
    ax.semilogy(k, errors, label=r"Погрешность $||x^{(k)}-x^*||_\infty")
    ax.semilogy(k, steps, label=r"Разность $||x^{(k)}-x^{(k-1)}||_\infty")
    ax.set_xlabel("Номер итерации")
    ax.set_ylabel("Норма вектора")
    ax.set_title("Метод Зейделя для нормальной системы 1")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_false_convergence_normal_system1.png", dpi=180)
    plt.close(fig)

    # Рисунок 6. Фактическое число итераций при различных требованиях к точности.
    epsilons = np.array([1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12])
    fig, ax = plt.subplots(figsize=(9, 6))
    plot_cases = [
        ("Система 1: простые итерации", np.round(inverse_gauss_jordan(A1), 2) @ A1,
         np.round(inverse_gauss_jordan(A1), 2) @ b1, "jacobi", "-"),
        ("Система 1: Зейдель", np.round(inverse_gauss_jordan(A1), 2) @ A1,
         np.round(inverse_gauss_jordan(A1), 2) @ b1, "seidel", "--"),
        ("Система 2: простые итерации", A2.T @ A2, A2.T @ b2, "jacobi", "-"),
        ("Система 2: Зейдель", A2.T @ A2, A2.T @ b2, "seidel", "--"),
    ]
    for label, matrix, rhs, method, style in plot_cases:
        b_iter, c_iter = to_iterative_form(matrix, rhs)
        counts = [count_iterations(b_iter, c_iter, method, eps) for eps in epsilons]
        ax.plot(-np.log10(epsilons), counts, marker="o", linestyle=style, label=label)
    ax.set_xlabel(r"$-\lg(\varepsilon)$")
    ax.set_ylabel("Число итераций")
    ax.set_title("Зависимость числа итераций от требуемой точности")
    ax.set_xticks(-np.log10(epsilons), [r"$10^{-2}$", r"$10^{-4}$", r"$10^{-6}$",
                                         r"$10^{-8}$", r"$10^{-10}$", r"$10^{-12}$"])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_iterations_vs_accuracy.png", dpi=180)
    plt.close(fig)

    print("\nСозданы шесть графиков в папке results:")
    for path in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  {path.name}")


def run_experiments() -> None:
    """Выполняет расчёты и строит только графики, включённые в отчёт."""
    np.set_printoptions(precision=8, suppress=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("ПРАКТИЧЕСКАЯ РАБОТА № 1. СЛАУ. ВАРИАНТ 12")
    print("=" * 76)
    print("\nЧАСТЬ 1. МЕТОД ГАУССА И ИССЛЕДОВАНИЕ ТОЧНОСТИ")
    solutions = {
        "Система 1": print_direct_method_results("A1", A1, b1),
        "Система 2": print_direct_method_results("A2", A2, b2),
    }

    print("\nЧАСТЬ 2. ИТЕРАЦИОННЫЕ МЕТОДЫ")
    for system_name, matrix, rhs, _ in build_transforms():
        b_matrix, c_vector = to_iterative_form(matrix, rhs)
        rho_jacobi = spectral_radius(b_matrix)
        rho_seidel = spectral_radius(seidel_iteration_matrix(b_matrix))
        print(f"\n{system_name}: rho(B)={rho_jacobi:.8g}; rho(T)={rho_seidel:.8g}")

    save_plots(solutions)
    print("\nРасчёт завершён.")
    print(f"Графики сохранены в: {OUTPUT_DIR.resolve()}")
