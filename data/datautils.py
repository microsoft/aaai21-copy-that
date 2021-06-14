import random
from typing import Iterator, TypeVar, Iterable, Callable


class LazyDataIterable(Iterable):
    def __init__(self, base_iterable_func: Callable[[], Iterator]):
        self.__base_iterable_func = base_iterable_func

    def __iter__(self):
        return self.__base_iterable_func()
