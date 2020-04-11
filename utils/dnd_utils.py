from collections import deque

import numpy as np


def roll_dice(sides):
    return np.random.random_integers(1, sides, 1)[0]


def calculate_distance(coord1, coord2):
    return np.sum(np.sqrt((coord1 - coord2)**2))


def draw_location(console, x, y, char="*"):
    console.addstr(y, x, char)
    console.refresh()


class TurnOrder(object):
    def __init__(self, items=()):
        self.deque = deque(items)

    def __iter__(self):
        return self

    def len(self):
        return len(self.deque)

    def __next__(self):
        if not self.deque:
            raise StopIteration
        item = self.deque.popleft()
        self.deque.append(item)
        return item

    next = __next__

    def delete_next(self):
        self.deque.popleft()

    def delete_prev(self):
        self.deque.pop()
