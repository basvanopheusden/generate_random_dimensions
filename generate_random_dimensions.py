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

    # Basic argument validation
    if num_dimensions < 0:
        raise ValueError("num_dimensions must be non-negative")
    if min_size < 1:
        raise ValueError("min_size must be at least 1")
    if max_size < min_size:
        raise ValueError("max_size must be greater than or equal to min_size")
    if total_elements is not None and total_elements < 1:
        raise ValueError("total_elements must be at least 1 if supplied")
    if gcd_constraint is not None and gcd_constraint < 1:
        raise ValueError("gcd_constraint must be at least 1 if supplied")

    if num_dimensions == 0:
        if gcd_constraint is not None:
            raise ValueError(
                "gcd_constraint cannot be specified when num_dimensions is 0"
            )
        if total_elements is not None and total_elements != 1:
            raise ValueError(
                "total_elements must be 1 when num_dimensions is 0"
            )
        return ()

    if gcd_constraint is not None:
        if gcd_constraint < min_size or gcd_constraint > max_size:
            raise ValueError(
                "gcd_constraint must fall within the size bounds in order to "
                "appear as a dimension"
            )

        if total_elements is not None:
            min_possible = gcd_constraint ** num_dimensions
            if total_elements < min_possible:
                raise ValueError(
                    "total_elements is too small to accommodate gcd_constraint"
                )
            if total_elements % (gcd_constraint ** num_dimensions) != 0:
                raise ValueError(
                    "total_elements is incompatible with gcd_constraint"
                )

    # Helper to search for all valid tuples using backtracking
    valid_shapes: List[Tuple[int, ...]] = []

    def _search(
        idx: int,
        partial: List[int],
        remaining: Optional[int],
        current_gcd: Optional[int],
        has_exact: bool,
    ) -> None:
        if idx == num_dimensions:
            if total_elements is not None and remaining != 1:
                return
            final_gcd = current_gcd if current_gcd is not None else 0
            if gcd_constraint is not None:
                if final_gcd != gcd_constraint or not has_exact:
                    return
            valid_shapes.append(tuple(partial))
            return

        for value in range(min_size, max_size + 1):
            if gcd_constraint is not None and value % gcd_constraint != 0:
                continue
            if total_elements is not None:
                if remaining is None or remaining % value != 0:
                    continue
                next_remaining = remaining // value
            else:
                next_remaining = None

            next_gcd = value if current_gcd is None else math.gcd(current_gcd, value)
            if gcd_constraint is not None and gcd_constraint % next_gcd != 0:
                continue

            partial.append(value)
            _search(idx + 1, partial, next_remaining, next_gcd, has_exact or (value == gcd_constraint))
            partial.pop()

    _search(
        0,
        [],
        total_elements if total_elements is not None else None,
        None,
        False,
    )

    if not valid_shapes:
        if total_elements is not None and gcd_constraint is not None:
            raise ValueError(
                "total_elements and gcd_constraint cannot be simultaneously satisfied"
            )
        raise ValueError("No tuple satisfying the requested constraints can be found")

    return rng.choice(valid_shapes)
