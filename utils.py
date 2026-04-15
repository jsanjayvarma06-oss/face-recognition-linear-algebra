"""utils.py — Console formatting helpers."""

STEP_LABELS = {
    1: "Matrix Representation",
    2: "Matrix Simplification",
    3: "Structure of the Space",
    4: "Remove Redundancy",
    5: "Orthogonalization",
    6: "Projection",
    7: "Prediction / Least Squares",
    8: "Pattern Discovery (Eigenvalues)",
    9: "System Simplification (Diagonalization)",
}

LINE = "─" * 70


def print_banner() -> None:
    print("\n" + "═" * 70)
    print("  PES UNIVERSITY  |  UE24MA241B — Linear Algebra & Its Applications")
    print("  Mini Project : Face Recognition Using Linear Algebra Pipeline")
    print("═" * 70 + "\n")


def print_section(title: str) -> None:
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}\n")


def print_step(n: int, title: str = "") -> None:
    label = title or STEP_LABELS.get(n, "")
    print(f"\n  ┌── STEP {n}: {label}")
    print(  "  │")
