from __future__ import annotations
from typing import Optional
import pygame
from .entities import Direction


class PygameRenderer:

    def __init__(self, cell_size: int = 32, fps: int = 60) -> None:
        self.cell_size = int(cell_size)
        self.fps = int(fps)
        self._screen: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None

    def init(self, w: int, h: int, title: str = "Tank RL") -> None:
        pygame.init()
        self._screen = pygame.display.set_mode((w * self.cell_size, h * self.cell_size))
        pygame.display.set_caption(title)
        self._clock = pygame.time.Clock()

    def close(self) -> None:
        pygame.quit()
        self._screen = None
        self._clock = None

    def render(self, env, text: str = "") -> None:
        if env.state is None:
            return

        if self._screen is None:
            self.init(env.w, env.h)

        assert self._screen is not None
        assert self._clock is not None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit

        cs = self.cell_size
        self._screen.fill((20, 20, 20))

        # walls
        for (wx, wy) in env.state.walls:
            pygame.draw.rect(self._screen, (70, 70, 70), (wx * cs, wy * cs, cs, cs))

        # target
        tx, ty = env.state.target.x, env.state.target.y
        pygame.draw.rect(self._screen, (200, 180, 60), (tx * cs, ty * cs, cs, cs))

        # tank
        x, y = env.state.tank.x, env.state.tank.y
        pygame.draw.rect(self._screen, (60, 160, 220), (x * cs, y * cs, cs, cs))

        # facing direction indicator
        cx, cy = x * cs + cs // 2, y * cs + cs // 2
        dx, dy = 0, 0

        if env.state.tank.dir == Direction.N:
            dy = -cs // 3
        elif env.state.tank.dir == Direction.E:
            dx = cs // 3
        elif env.state.tank.dir == Direction.S:
            dy = cs // 3
        elif env.state.tank.dir == Direction.W:
            dx = -cs // 3

        pygame.draw.line(self._screen, (255, 255, 255), (cx, cy), (cx + dx, cy + dy), 3)

        shot = getattr(env, "last_shot", None)
        ttl = getattr(env, "last_shot_ttl", 0)

        if shot is not None and ttl > 0:
            x0, y0, x1, y1, hit = shot

            def center(px: int, py: int) -> tuple[int, int]:
                return (px * cs + cs // 2, py * cs + cs // 2)

            p0 = center(x0, y0)
            p1 = center(x1, y1)
            color = (60, 220, 120) if hit else (220, 80, 80)
            pygame.draw.line(self._screen, color, p0, p1, 4)

        if text:
            font = pygame.font.SysFont(None, 20)
            surf = font.render(text, True, (230, 230, 230))
            self._screen.blit(surf, (6, 6))

        pygame.display.flip()
        self._clock.tick(self.fps)