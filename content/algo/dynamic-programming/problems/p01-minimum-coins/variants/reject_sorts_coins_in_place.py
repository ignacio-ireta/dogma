from __future__ import annotations


def min_coins(coins: list[int], amount: int) -> int:
    """Wrong on purpose: sorts the caller's list in place. Return the fewest coins that sum to amount, or -1 if impossible.

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
    """  # noqa: E501
    # Validate first. type(...) is int rejects bool, which is a subclass of int.
    if type(coins) is not list:
        raise TypeError(f"coins must be a list, got {type(coins).__name__}")
    for coin in coins:
        if type(coin) is not int:
            raise TypeError(f"every coin must be a Python int, got {type(coin).__name__}")
        if coin < 1:
            raise ValueError("every coin must be at least 1")
    if type(amount) is not int:
        raise TypeError(f"amount must be a Python int, got {type(amount).__name__}")
    if amount < 0:
        raise ValueError("amount must be at least 0")

    coins.sort()

    # The state is the value v. best[v] is the fewest coins that make v exactly.
    # Base case: best[0] = 0, because zero coins make zero.
    # Impossible values start at INF = amount + 1. No achievable count for v <= amount can
    # reach that, because a count never exceeds v (each coin is at least 1). Using an integer
    # keeps the table ints, and no addition can make it look smaller than a real count.
    inf = amount + 1
    best = [0] + [inf] * amount

    # Order of evaluation: best[v] reads best[v - coin], which is smaller than v. Increasing v
    # guarantees every read is already final.
    for v in range(1, amount + 1):
        for coin in coins:
            # Transition: use one coin of value coin, then make the remaining v - coin.
            if coin <= v and best[v - coin] + 1 < best[v]:
                best[v] = best[v - coin] + 1

    # Answer location: the table entry at amount. If it is still inf, the amount is impossible.
    return best[amount] if best[amount] <= amount else -1
