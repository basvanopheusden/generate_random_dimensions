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
    # ---------------------------------------------------------------------
    # 0.  Basic validation and RNG selection
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # 1.  Trivial 0-dimension case
    # ---------------------------------------------------------------------
    if num_dimensions == 0:
        if total_elements is not None and total_elements != 1:
            raise ValueError("total_elements must be 1 when num_dimensions is 0")
        if gcd_constraint is not None:
            raise ValueError("gcd_constraint makes no sense for 0 dimensions")
        return ()

    # ---------------------------------------------------------------------
    # 2.  Helper(s)
    # ---------------------------------------------------------------------
    def _gcd_of_list(values: List[int]) -> int:
        g = 0
        for v in values:
            g = math.gcd(g, v)
        return g

    # Memo for the (expensive) recursive divisor search used when a product
    # constraint is present.
    _memo: Dict[Tuple[int, int], Optional[List[int]]] = {}

    def _find_factors(
        product: int,
        dims_left: int,
        low: int,
        high: int,
    ) -> Optional[List[int]]:
        key = (product, dims_left)
        if key in _memo:
            return _memo[key]

        # Base-case – only one dimension left → must equal the remaining
        # product *and* satisfy the bounds.
        if dims_left == 1:
            if low <= product <= high:
                return [product]
            _memo[key] = None
            return None

        # Early impossibility pruning:  check whether the remaining
        # product is even theoretically reachable with the bounds.
        if product < low**dims_left or product > high**dims_left:
            _memo[key] = None
            return None

        # Enumerate *divisors* of the current product that are within range
        # and that leave a residue still factorable by the remaining
        # dimensions.
        #
        # Instead of generating **all** divisors (which might be expensive),
        # we simply iterate through the admissible numbers in random order
        # and keep the ones that divide *product*.
        possible = [d for d in range(low, high + 1) if product % d == 0]
        rng.shuffle(possible)

        for d in possible:
            sub = _find_factors(product // d, dims_left - 1, low, high)
            if sub is not None:
                _memo[key] = [d] + sub
                return _memo[key]

        _memo[key] = None
        return None

    # ---------------------------------------------------------------------
    # 3.  Scenario A – no total_elements and no gcd_constraint
    # ---------------------------------------------------------------------
    if total_elements is None and gcd_constraint is None:
        return tuple(rng.randint(min_size, max_size) for _ in range(num_dimensions))

    # ---------------------------------------------------------------------
    # 4.  Scenarios where *gcd_constraint* is given
    # ---------------------------------------------------------------------
    if gcd_constraint is not None:
        g = gcd_constraint

        if not (min_size <= g <= max_size):
            raise ValueError("At least one dimension must be able to equal gcd_constraint")

        # All dimensions are multiples of *g*.  Work with the scaled-down
        # variables  k_i  such that  d_i = g * k_i.
        k_min = (min_size + g - 1) // g  # ⌈min_size / g⌉
        k_max = max_size // g  # ⌊max_size / g⌋

        if k_min > k_max:
            raise ValueError("No multiple of gcd_constraint fits into size bounds")

        # -----------------------------------------------------------------
        # 4a.  Only gcd_constraint (no total_elements)
        # -----------------------------------------------------------------
        if total_elements is None:
            # Single-dimension shape is trivial.
            if num_dimensions == 1:
                return (g,)

            # We *must* be able to use k_i = 1, otherwise the “one dimension
            # equals g” requirement can never be met.
            if k_min > 1:
                raise ValueError("Size bounds prevent any dimension equalling the gcd")

            # Build a shape where the first dimension is exactly *g*.  The
            # remaining k_i values are drawn randomly until the gcd(k_i) is 1
            # (which is automatically true because one of them is 1, but we
            # keep the loop for completeness and in case k_min == k_max == 1).
            RETRIES = 1_000
            for _ in range(RETRIES):
                ks = [1] + [rng.randint(k_min, k_max) for _ in range(num_dimensions - 1)]
                if _gcd_of_list(ks) == 1:
                    dims = [g * k for k in ks]
                    rng.shuffle(dims)
                    return tuple(dims)
            raise ValueError("Unable to construct shape satisfying the gcd constraint")

        # -----------------------------------------------------------------
        # 4b.  Both total_elements *and* gcd_constraint
        # -----------------------------------------------------------------
        # total_elements must equal g^n · Πk_i
        try:
            g_pow_n = g**num_dimensions
        except OverflowError as exc:  # pragma: no cover – would need gigantic numbers
            raise ValueError("gcd_constraint and num_dimensions are too large") from exc

        if total_elements % g_pow_n != 0:
            raise ValueError("total_elements must be divisible by gcd_constraint ** num_dimensions")

        k_product = total_elements // g_pow_n

        # Single-dimension again trivial.
        if num_dimensions == 1:
            if k_product != 1:
                raise ValueError("No valid single-dimension shape exists for the given product and gcd")
            return (g,)

        # If k_min > 1 we cannot place a k_i = 1 → impossible.
        if k_min > 1:
            raise ValueError("Size bounds prevent any dimension equalling the gcd")

        # Try to fix one k_i as 1 (thus one dimension equals g) and factor
        # the remaining product over the remaining dimensions.
        #
        # NOTE:  The gcd will automatically be g because gcd(1, …) == 1.
        remaining_dims = num_dimensions - 1
        factors = _find_factors(k_product, remaining_dims, k_min, k_max)
        if factors is None:
            raise ValueError("No shape satisfies the combined product/gcd constraints")

        ks = [1] + factors
        dims = [g * k for k in ks]
        rng.shuffle(dims)
        return tuple(dims)

    # ---------------------------------------------------------------------
    # 5.  Scenario – total_elements but *no* gcd_constraint
    # ---------------------------------------------------------------------
    else:
        # num_dimensions == 1 has a very simple solution space
        if num_dimensions == 1:
            if min_size <= total_elements <= max_size:
                return (total_elements,)
            raise ValueError("total_elements is outside the admissible size bounds for 1D")

        # Early range-check to avoid hopeless searches
        if total_elements < min_size**num_dimensions or total_elements > max_size**num_dimensions:
            raise ValueError("total_elements cannot be realised with the given size bounds")

        factors = _find_factors(total_elements, num_dimensions, min_size, max_size)
        if factors is None:
            raise ValueError("No shape satisfies the given total_elements and size bounds")

        rng.shuffle(factors)
        return tuple(factors)
