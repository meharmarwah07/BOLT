"""Tests for stats.wilson(). Run with: python test_stats.py"""
import sys

from stats import wilson


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


TESTS = [
    ("known interval (50/100) ~ (0.404, 0.596)", test_known_interval),
    ("zero successes (0/20)", test_zero_successes),
    ("all successes (20/20)", test_all_successes),
    ("interval always within [0, 1]", test_bounds_within_0_1),
    ("interval narrows as n grows (fixed ratio)", test_narrows_with_n),
    ("bad inputs raise ValueError", test_bad_inputs_raise),
]


if __name__ == "__main__":
    results = [case(name, fn) for name, fn in TESTS]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)
