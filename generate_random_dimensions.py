import math
import random
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple


def generate_random_dimensions(
    num_dimensions: int,
    *,
    min_size: int = 1,
    max_size: int = 10,
    total_elements: Optional[int] = None,
    gcd_constraint: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[int, ...]:
    """
    Produce a random shape (a tuple of positive integers) that can be used to
    initialise a multi-dimensional array or tensor, while satisfying optional
    arithmetic constraints on the shape.

    Inputs
    ------
    num_dimensions : int
        Number of axes to generate. If 0, the function returns an empty tuple.
    min_size : int
        Inclusive lower bound for every dimension. Default is 1.
    max_size : int
        Inclusive upper bound for every dimension. Must be greater than or
        equal to min_size. Default is 10.
    total_elements : Optional[int]
        If provided, the product of all returned dimensions must be exactly
        this value. Default is None.
    gcd_constraint : Optional[int]
        If provided, the greatest common divisor of all returned dimensions
        must be exactly this value. Equivalently, every dimension must be
        a multiple of gcd_constraint and at least one dimension must equal
        gcd_constraint. Default is None.
    rng : Optional[random.Random]
        The random number generator to be used. If None (default), the
        global generator in the random module is used. Supplying an explicit
        random.Random instance allows callers to obtain deterministic
        output by seeding that instance themselves.

    Outputs
    -------
    Tuple[int, ...]
        A tuple of length num_dimensions such that:
        - Every element is between min_size and max_size (inclusive)
        - The product of the elements equals total_elements if that
          argument is not None
        - The greatest common divisor of the elements equals
          gcd_constraint if that argument is not None

    Raises
    ------
    ValueError
        If num_dimensions is negative.
    ValueError
        If min_size is less than 1.
    ValueError
        If max_size is less than min_size.
    ValueError
        If total_elements is not None and is less than 1.
    ValueError
        If gcd_constraint is not None and is less than 1.
    ValueError
        If both total_elements and gcd_constraint are supplied but they
        cannot be simultaneously satisfied by any tuple that also respects
        the size bounds.
    ValueError
        If no tuple satisfying the requested constraints can be found.

    Assumptions
    -----------
    The caller never relies on any specific ordering of the returned
    dimensions—the tuple may be in any order chosen by the implementation.

    When num_dimensions is 0, the only admissible value for total_elements
    (if supplied) is 1, because the product of an empty set of numbers is
    conventionally one.

    When multiple valid solutions exist that satisfy all constraints, the
    function selects one uniformly at random (or pseudo-randomly if an rng
    is provided).
    """

    # ------------------------------------------------------------
    # 0. Parameter validation and RNG setup
    # ------------------------------------------------------------
    if num_dimensions < 0:
        raise ValueError("num_dimensions cannot be negative")
    if min_size < 1:
        raise ValueError("min_size must be at least 1")
    if max_size < min_size:
        raise ValueError("max_size must be >= min_size")
    if total_elements is not None and total_elements < 1:
        raise ValueError("total_elements must be at least 1")
    if gcd_constraint is not None and gcd_constraint < 1:
        raise ValueError("gcd_constraint must be at least 1")

    rng = rng or random

    # ------------------------------------------------------------
    # 1. Zero-dimension special case
    # ------------------------------------------------------------
    if num_dimensions == 0:
        if total_elements is not None and total_elements != 1:
            raise ValueError("total_elements must be 1 when num_dimensions is 0")
        if gcd_constraint is not None:
            raise ValueError("gcd_constraint makes no sense for 0 dimensions")
        return ()

    # ------------------------------------------------------------
    # 2. Trivial scenario: no arithmetic constraints
    # ------------------------------------------------------------
    if total_elements is None and gcd_constraint is None:
        return tuple(rng.randint(min_size, max_size) for _ in range(num_dimensions))

    # ------------------------------------------------------------
    # 3. Determine admissible values for each dimension
    # ------------------------------------------------------------
    if gcd_constraint is None:
        admissible = list(range(min_size, max_size + 1))
    else:
        admissible = [v for v in range(min_size, max_size + 1) if v % gcd_constraint == 0]
        if not admissible:
            raise ValueError("No multiple of gcd_constraint fits into size bounds")
        if gcd_constraint not in admissible:
            raise ValueError("At least one dimension must be able to equal gcd_constraint")
        if total_elements is not None:
            try:
                g_pow = gcd_constraint ** num_dimensions
            except OverflowError as exc:  # pragma: no cover - extremely large numbers
                raise ValueError("gcd_constraint and num_dimensions are too large") from exc
            if total_elements % g_pow != 0:
                raise ValueError(
                    "total_elements must be divisible by gcd_constraint ** num_dimensions"
                )

    if total_elements is not None:
        min_possible = (min(admissible)) ** num_dimensions
        max_possible = (max(admissible)) ** num_dimensions
        if total_elements < min_possible or total_elements > max_possible:
            raise ValueError("total_elements cannot be realised with the given size bounds")

    # ------------------------------------------------------------
    # 4. Enumerate all valid shapes
    # ------------------------------------------------------------
    solutions: List[Tuple[int, ...]] = []

    def _search(index: int, dims: List[int], prod: int, cur_gcd: int, has_g: bool) -> None:
        """Recursive helper that collects all admissible shapes."""

        if index == num_dimensions:
            if total_elements is not None and prod != total_elements:
                return
            if gcd_constraint is not None:
                if cur_gcd != gcd_constraint or not has_g:
                    return
            solutions.append(tuple(dims))
            return

        remaining = num_dimensions - index - 1

        for val in admissible:
            new_prod = prod * val

            if total_elements is not None:
                if new_prod > total_elements:
                    continue
                if remaining:
                    if total_elements % new_prod != 0:
                        continue
                    # theoretical min/max products with remaining dims
                    min_rest = (min(admissible)) ** remaining
                    max_rest = (max(admissible)) ** remaining
                    if new_prod * min_rest > total_elements:
                        continue
                    if new_prod * max_rest < total_elements:
                        continue

            new_gcd = val if index == 0 else math.gcd(cur_gcd, val)

            dims.append(val)
            _search(index + 1, dims, new_prod, new_gcd, has_g or val == gcd_constraint)
            dims.pop()

    _search(0, [], 1, 0, False)

    if not solutions:
        raise ValueError("No tuple satisfying the requested constraints can be found.")

    choice = rng.choice(solutions)
    shuffled = list(choice)
    rng.shuffle(shuffled)
    return tuple(shuffled)

