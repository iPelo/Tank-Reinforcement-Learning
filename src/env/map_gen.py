from __future__ import annotations
from typing import Iterable,  Set, Tuple

Coord = Tuple[int, int]

def generate_walls  (
        w: int,
        h: int,
        density: float, rng,
        forbidden: Iterable[Coord] =(), ) -> Iterable[Coord]:
    forbidden_set = set(forbidden)
    walls: Set[Coord] = set()

    if density < 0.0:
        density = 0.0
    if density > 0.5:
        density = 0.5

    for y in range(h):
        for x in range(w):
            if (x,y) in forbidden_set:
                continue
            if x == 0 or y == 0 or x == w-1 or y == h-1:
                continue
            if rng.random() < density:
                walls.add((x,y))
    return walls

