"""Tests for stats.wilson() and stats.mcnemar_exact(). Run with: python test_stats.py"""
import sys

from stats import mcnemar_exact, wilson


def case(name, fn):
    try:
        fn()
        print(f"PASS {name}")
        return True
    except AssertionError as e:
        print(f"FAIL {name} ({e})")
        return False
    except Exception as e:
        print(f"FAIL {name} (unexpected {type(e).__name__}: {e})")
        return False


def test_known_interval():
    low, high = wilson(50, 100)
    assert round(low, 3) == 0.404, f"low={low}"
    assert round(high, 3) == 0.596, f"high={high}"


def test_zero_successes():
    low, high = wilson(0, 20)
    assert low == 0.0, f"low={low}"
    assert 0.12 <= high <= 0.17, f"high={high}"


def test_all_successes():
    low, high = wilson(20, 20)
    assert high == 1.0, f"high={high}"


def test_bounds_within_0_1():
    for k, n in [(0, 1), (1, 1), (5, 10), (1, 1000), (999, 1000), (0, 1000), (1000, 1000)]:
        low, high = wilson(k, n)
        assert 0.0 <= low <= 1.0, f"low={low} out of [0,1] for k={k}, n={n}"
        assert 0.0 <= high <= 1.0, f"high={high} out of [0,1] for k={k}, n={n}"
        assert low <= high, f"low>high for k={k}, n={n}"


def test_narrows_with_n():
    prev_width = None
    for n in (10, 100, 1000, 10000):
        k = n // 2
        low, high = wilson(k, n)
        width = high - low
        if prev_width is not None:
            assert width < prev_width, f"width did not narrow at n={n}: {width} >= {prev_width}"
        prev_width = width


def test_bad_inputs_raise():
    for k, n in [(0, 0), (5, -3), (-1, 10), (11, 10)]:
        try:
            wilson(k, n)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for k={k}, n={n}")


def test_mcnemar_known_value():
    # Matches R's binom.test(1, 10, 0.5)$p.value == 0.021484375 exactly --
    # exact McNemar on the discordant pairs reduces to this binomial test.
    p = mcnemar_exact(1, 9)
    assert round(p, 6) == round(0.021484375, 6), f"p={p}"


def test_mcnemar_symmetric_in_b_c():
    assert mcnemar_exact(2, 8) == mcnemar_exact(8, 2)


def test_mcnemar_no_discordant_pairs_is_1():
    assert mcnemar_exact(0, 0) == 1.0


def test_mcnemar_equal_split_is_1():
    # b == c (an even split of discordant pairs) is the least surprising
    # possible outcome under the null -- p must be exactly 1.0, not >1.
    for n in (2, 4, 10, 20):
        assert mcnemar_exact(n // 2, n // 2) == 1.0, f"n={n}"


def test_mcnemar_more_lopsided_is_more_significant():
    p_mild = mcnemar_exact(3, 7)
    p_extreme = mcnemar_exact(0, 10)
    assert p_extreme < p_mild


def test_mcnemar_bounds_within_0_1():
    for b, c in [(0, 0), (0, 1), (1, 0), (5, 5), (0, 50), (25, 25)]:
        p = mcnemar_exact(b, c)
        assert 0.0 <= p <= 1.0, f"p={p} out of [0,1] for b={b}, c={c}"


def test_mcnemar_bad_inputs_raise():
    for b, c in [(-1, 5), (5, -1)]:
        try:
            mcnemar_exact(b, c)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for b={b}, c={c}")


TESTS = [
    ("known interval (50/100) ~ (0.404, 0.596)", test_known_interval),
    ("zero successes (0/20)", test_zero_successes),
    ("all successes (20/20)", test_all_successes),
    ("interval always within [0, 1]", test_bounds_within_0_1),
    ("interval narrows as n grows (fixed ratio)", test_narrows_with_n),
    ("bad inputs raise ValueError", test_bad_inputs_raise),
    ("mcnemar_exact matches a known binom.test value", test_mcnemar_known_value),
    ("mcnemar_exact is symmetric in (b, c)", test_mcnemar_symmetric_in_b_c),
    ("mcnemar_exact(0, 0) == 1.0 (no discordant pairs)", test_mcnemar_no_discordant_pairs_is_1),
    ("mcnemar_exact(n/2, n/2) == 1.0 (even split)", test_mcnemar_equal_split_is_1),
    ("mcnemar_exact p shrinks as the split gets more lopsided", test_mcnemar_more_lopsided_is_more_significant),
    ("mcnemar_exact p always within [0, 1]", test_mcnemar_bounds_within_0_1),
    ("mcnemar_exact bad inputs raise ValueError", test_mcnemar_bad_inputs_raise),
]


if __name__ == "__main__":
    results = [case(name, fn) for name, fn in TESTS]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)
