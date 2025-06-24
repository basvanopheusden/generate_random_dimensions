import math
import random
from functools import reduce

import pytest


@pytest.mark.core
def test_total_elements_forcing_specific_result() -> None:
    # Only (2, 2) satisfies: two dims, each 2, product 4, bounds fixed at 2
    output = generate_random_dimensions(num_dimensions=2, total_elements=4, min_size=2, max_size=2)
    assert output == (2, 2)


@pytest.mark.core
def test_with_both_constraints_deterministic_result() -> None:
    # All dims forced to 6, so product is 216 and gcd is 6
    output = generate_random_dimensions(num_dimensions=3, total_elements=216, gcd_constraint=6, min_size=6, max_size=6)
    assert output == (6, 6, 6)


@pytest.mark.core
def test_large_num_dimensions() -> None:
    # Seeded RNG but any result must have 10 dims, each 1–2
    rng = random.Random(8)
    output = generate_random_dimensions(num_dimensions=10, min_size=1, max_size=2, rng=rng)
    assert len(output) == 10 and all(1 <= x <= 2 for x in output)


@pytest.mark.edge
def test_zero_dimensions() -> None:
    # With 0 dimensions the shape must be empty
    assert generate_random_dimensions(num_dimensions=0) == ()


@pytest.mark.edge
def test_zero_dimensions_with_total_elements_one() -> None:
    # total_elements=1 is the only admissible value when num_dimensions=0
    assert generate_random_dimensions(num_dimensions=0, total_elements=1) == ()


@pytest.mark.core
def test_with_gcd_constraint_equal_to_min() -> None:
    # Every dim is multiple of 5 within 5–15 and gcd is exactly 5
    rng = random.Random(6)
    output = generate_random_dimensions(num_dimensions=2, gcd_constraint=5, min_size=5, max_size=15, rng=rng)
    g = reduce(math.gcd, output)
    assert all(5 <= x <= 15 and x % 5 == 0 for x in output) and g == 5 and 5 in output


@pytest.mark.edge
def test_unsatisfiable_combination_raises() -> None:
    # With two dimensions confined to [5, 20], each must be a multiple of 5
    # and at least one dimension must equal 5.  The only multiples of 5 in
    # range are {5, 10, 15, 20}.  Exhaustive checking shows no pair drawn
    # from this set whose product is 150 **and** contains a 5.  Therefore,
    # the constraints are mutually inconsistent and the function must raise.
    rng = random.Random(42)
    with pytest.raises(ValueError):
        generate_random_dimensions(
            num_dimensions=2,
            min_size=5,
            max_size=20,
            total_elements=150,
            gcd_constraint=5,
            rng=rng,
        )


@pytest.mark.core
def test_custom_min_max_size() -> None:
    # Four dimensions, each between 5 and 8 inclusive
    rng = random.Random(3)
    output = generate_random_dimensions(num_dimensions=4, min_size=5, max_size=8, rng=rng)
    assert len(output) == 4 and all(5 <= x <= 8 for x in output)


@pytest.mark.edge
def test_with_prime_total_elements() -> None:
    # Product must be 17; only (1,17) or (17,1) possible within max_size=20
    rng = random.Random(42)
    output = generate_random_dimensions(num_dimensions=2, total_elements=17, max_size=20, rng=rng)
    assert math.prod(output) == 17 and set(output) == {1, 17}


@pytest.mark.core
def test_single_dimension() -> None:
    # Single dimension within default bounds 1–10
    rng = random.Random(2)
    output = generate_random_dimensions(num_dimensions=1, rng=rng)
    assert len(output) == 1 and 1 <= output[0] <= 10


@pytest.mark.edge
def test_total_elements_one_dim_error() -> None:
    # total_elements=42 exceeds default max_size=10, so no valid shape exists
    with pytest.raises(ValueError):
        generate_random_dimensions(num_dimensions=1, total_elements=42)


@pytest.mark.core
def test_basic_case_deterministic() -> None:
    # Three dimensions within default bounds produced by seeded RNG
    rng = random.Random(1)
    output = generate_random_dimensions(num_dimensions=3, rng=rng)
    assert len(output) == 3 and all(1 <= x <= 10 for x in output)


@pytest.mark.core
def test_with_gcd_constraint() -> None:
    # All dims multiples of 3 within 3–12, gcd exactly 3
    rng = random.Random(5)
    output = generate_random_dimensions(num_dimensions=3, gcd_constraint=3, min_size=3, max_size=12, rng=rng)
    g = reduce(math.gcd, output)
    assert all(3 <= x <= 12 and x % 3 == 0 for x in output) and g == 3 and 3 in output


@pytest.mark.core
def test_equal_min_max_size() -> None:
    # min_size=max_size=5 forces every dim to be 5
    assert generate_random_dimensions(num_dimensions=3, min_size=5, max_size=5) == (5, 5, 5)


@pytest.mark.core
def test_with_total_elements() -> None:
    # Two dims whose product is 100 and each ≤ 20
    rng = random.Random(4)
    output = generate_random_dimensions(num_dimensions=2, max_size=20, total_elements=100, rng=rng)
    assert math.prod(output) == 100 and all(1 <= x <= 20 for x in output)

