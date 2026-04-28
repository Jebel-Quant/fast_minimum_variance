"""Main module for fast_minimum_variance."""


def say_hello(name: str) -> str:
    """Say hello to the user.

    Args:
        name: The name of the user.

    Returns:
        A greeting string.
    """
    return f"Hello, {name}!"


def main() -> None:
    """Execute the main function."""
    print(say_hello("World"))