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

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    if num_dimensions < 0:
        raise ValueError("num_dimensions must be non-negative")

    if min_size < 1:
        raise ValueError("min_size must be at least 1")

    if max_size < min_size:
        raise ValueError("max_size must be greater than or equal to min_size")

    if total_elements is not None and total_elements < 1:
        raise ValueError("total_elements must be at least 1 if provided")

    if gcd_constraint is not None and gcd_constraint < 1:
        raise ValueError("gcd_constraint must be at least 1 if provided")

    rng = rng or random

    # Special case: zero dimensions
    if num_dimensions == 0:
        if total_elements is not None and total_elements != 1:
            raise ValueError("total_elements must be 1 when num_dimensions is 0")
        if gcd_constraint is not None:
            raise ValueError(
                "gcd_constraint cannot be satisfied when num_dimensions is 0"
            )
        return ()

    # Candidate values respecting the basic size bounds and gcd constraint
    if gcd_constraint is not None:
        candidates = [
            v for v in range(min_size, max_size + 1) if v % gcd_constraint == 0
        ]
        if not candidates:
            raise ValueError(
                "No multiples of gcd_constraint fall within the allowed size range"
            )
    else:
        candidates = list(range(min_size, max_size + 1))

    # Quick infeasibility checks when both total_elements and gcd_constraint
    # are provided. These checks only cover obvious contradictions.
    if total_elements is not None and gcd_constraint is not None:
        if gcd_constraint not in candidates:
            raise ValueError(
                "gcd_constraint is outside the allowed size range"
            )
        if total_elements % gcd_constraint != 0:
            raise ValueError(
                "total_elements is not divisible by gcd_constraint"
            )
        if gcd_constraint ** num_dimensions > total_elements:
            raise ValueError(
                "Constraints on total_elements and gcd_constraint cannot be satisfied"
            )

    # Pre-compute some helper values for pruning during search
    min_candidate = min(candidates)
    max_candidate = max(candidates)

    results: List[Tuple[int, ...]] = []
    current: List[int] = []

    def dfs(index: int, product: int, current_gcd: int, used_gcd_val: bool) -> None:
        """Depth-first search for valid tuples."""
        if index == num_dimensions:
            if total_elements is not None and product != total_elements:
                return
            if gcd_constraint is not None:
                if current_gcd != gcd_constraint or not used_gcd_val:
                    return
            results.append(tuple(current))
            return

        remaining = num_dimensions - index - 1
        for value in candidates:
            new_product = product * value
            if total_elements is not None:
                if new_product > total_elements:
                    continue
                if total_elements % new_product != 0 and remaining == 0:
                    continue
                # Bound the remaining product to prune impossible branches
                if remaining > 0:
                    if new_product * (min_candidate ** remaining) > total_elements:
                        continue
                    if new_product * (max_candidate ** remaining) < total_elements:
                        continue

            new_gcd = math.gcd(current_gcd, value) if index > 0 else value
            if gcd_constraint is not None:
                if new_gcd < gcd_constraint or new_gcd % gcd_constraint != 0:
                    continue

            current.append(value)
            dfs(index + 1, new_product, new_gcd, used_gcd_val or value == gcd_constraint)
            current.pop()

    dfs(0, 1, 0, False)

    if not results:
        if total_elements is not None and gcd_constraint is not None:
            raise ValueError(
                "total_elements and gcd_constraint cannot be simultaneously satisfied"
            )
        raise ValueError("No tuple satisfying the requested constraints can be found")

    return rng.choice(results)
