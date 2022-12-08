from builtins import __namedtuple
from typing import Sequence, Iterable, Any
# types: Sequences(list/tuple)(aka ordered sets), Iterable (lists, tuples, sets
# , dictionaries, strings, etc)(anything you can iterate over)


def three_length_words(words: list) -> list:
    return [word for word in words if len(word) >= 3]


print(three_length_words(['There', 'has', 'never', 'been', 'a', 'sadness', 'that', 'can’t', 'be', 'cured', 'by', 'breakfast', 'food', '.']))


class Pointer:
    def __init__(self, *args):
        if len(args) == 2:
            if not isinstance(args[0], int) or not isinstance(args[1], int):
                raise TypeError("Invalid parameters")
            self.x = args[0]
            self.y = args[1]
        elif len(args) == 0:
            self.x = 0
            self.y = 0
        else:
            raise Exception("Invalid parameters")

    def get_position(self):
        return self.x, self.y

    def set_x(self, x):
        self.x = x

    def set_y(self, y):
        self.y = y


pointer = Pointer(1, 1)
print(pointer.get_position())
pointer.set_x(5)
print(pointer.get_position())


