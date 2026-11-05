from __future__ import annotations


def min_coins(coins: list[int], amount: int) -> int:
    """Return the fewest coins (with repeats) that sum to amount, or -1 if impossible.

    Parameters
    ----------
    coins : list of Python int, possibly empty, each at least 1. Order is arbitrary and values
        may repeat. Must not be mutated.
    amount : Python int, at least 0.

    Returns
    -------
    int : the minimum number of coins summing to amount; 0 when amount is 0; -1 when no
        combination of the coins makes amount.
        Runs in O(amount * len(coins)) time.

    Raises
    ------
    TypeError
        When coins is not a list, a coin is not a Python int, or amount is not a Python int.
    ValueError
        When a coin is less than 1, or amount is negative.
    """
    raise NotImplementedError
