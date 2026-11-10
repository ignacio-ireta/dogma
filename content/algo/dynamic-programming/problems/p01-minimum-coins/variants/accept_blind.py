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
    if not isinstance(coins, list):
        raise TypeError("coins must be a list")
    for coin in coins:
        if type(coin) is not int:
            raise TypeError("each coin must be a Python int")
        if coin < 1:
            raise ValueError("each coin must be at least 1")
    if type(amount) is not int:
        raise TypeError("amount must be a Python int")
    if amount < 0:
        raise ValueError("amount must be non-negative")

    unreachable = amount + 1
    best = [0] + [unreachable] * amount
    for value in range(1, amount + 1):
        fewest = unreachable
        for coin in coins:
            if coin <= value and best[value - coin] + 1 < fewest:
                fewest = best[value - coin] + 1
        best[value] = fewest
    return -1 if best[amount] == unreachable else best[amount]
