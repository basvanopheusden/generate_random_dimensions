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
    if rng is None:
        rng = random

    # Validate arguments
    if num_dimensions < 0:
        raise ValueError("num_dimensions is negative")
    if min_size < 1:
        raise ValueError("min_size is less than 1")
    if max_size < min_size:
        raise ValueError("max_size is less than min_size")
    if total_elements is not None and total_elements < 1:
        raise ValueError("total_elements is not None and is less than 1")
    if gcd_constraint is not None and gcd_constraint < 1:
        raise ValueError("gcd_constraint is not None and is less than 1")

    # Handle the trivial zero-dimension case
    if num_dimensions == 0:
        if total_elements is not None and total_elements != 1:
            raise ValueError(
                "No tuple satisfying the requested constraints can be found."
            )
        if gcd_constraint is not None:
            raise ValueError(
                "No tuple satisfying the requested constraints can be found."
            )
        return ()

    # Pre-compute the set of admissible values for each dimension
    if gcd_constraint is not None:
        if gcd_constraint < min_size or gcd_constraint > max_size:
            raise ValueError(
                "No tuple satisfying the requested constraints can be found."
            )
        admissible_values = [
            v for v in range(min_size, max_size + 1) if v % gcd_constraint == 0
        ]
    else:
        admissible_values = list(range(min_size, max_size + 1))

    if not admissible_values:
        raise ValueError("No tuple satisfying the requested constraints can be found.")

    min_candidate = min(admissible_values)
    max_candidate = max(admissible_values)

    if total_elements is not None:
        # Quick bounds check on the total product
        if (
            total_elements < min_candidate ** num_dimensions
            or total_elements > max_candidate ** num_dimensions
        ):
            if gcd_constraint is not None:
                raise ValueError(
                    "total_elements and gcd_constraint cannot be simultaneously satisfied by any tuple that also respects the size bounds."
                )
            raise ValueError(
                "No tuple satisfying the requested constraints can be found."
            )
        if gcd_constraint is not None:
            base = gcd_constraint ** num_dimensions
            if total_elements % base != 0:
                raise ValueError(
                    "total_elements and gcd_constraint cannot be simultaneously satisfied by any tuple that also respects the size bounds."
                )

    valid: List[Tuple[int, ...]] = []

    def backtrack(
        idx: int,
        current: List[int],
        prod: int,
        gcd_so_far: int,
        used_exact: bool,
    ) -> None:
        if idx == num_dimensions:
            if total_elements is not None and prod != total_elements:
                return
            if gcd_constraint is not None:
                if gcd_so_far != gcd_constraint or not used_exact:
                    return
            valid.append(tuple(current))
            return

        remaining = num_dimensions - idx - 1

        for value in admissible_values:
            new_prod = prod * value
            if total_elements is not None:
                if new_prod > total_elements:
                    continue
                if remaining > 0:
                    min_possible = new_prod * (min_candidate ** remaining)
                    max_possible = new_prod * (max_candidate ** remaining)
                    if min_possible > total_elements or max_possible < total_elements:
                        continue

            new_gcd = value if idx == 0 else math.gcd(gcd_so_far, value)
            backtrack(
                idx + 1,
                current + [value],
                new_prod,
                new_gcd,
                used_exact or (gcd_constraint is not None and value == gcd_constraint),
            )

    backtrack(0, [], 1, 0, False)

    if not valid:
        if total_elements is not None and gcd_constraint is not None:
            raise ValueError(
                "total_elements and gcd_constraint cannot be simultaneously satisfied by any tuple that also respects the size bounds."
            )
        raise ValueError("No tuple satisfying the requested constraints can be found.")

    return rng.choice(valid)
