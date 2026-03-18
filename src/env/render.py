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

        def draw_tank(tank, body_color):
            x, y = tank.x, tank.y
            pygame.draw.rect(self._screen, body_color, (x * cs, y * cs, cs, cs))

            cx, cy = x * cs + cs // 2, y * cs + cs // 2
            dx, dy = 0, 0

            if tank.dir == Direction.N:
                dy = -cs // 3
            elif tank.dir == Direction.E:
                dx = cs // 3
            elif tank.dir == Direction.S:
                dy = cs // 3
            elif tank.dir == Direction.W:
                dx = -cs // 3

            pygame.draw.line(self._screen, (255, 255, 255), (cx, cy), (cx + dx, cy + dy), 3)

        draw_tank(env.state.player, (60, 160, 220))
        draw_tank(env.state.enemy, (220, 120, 60))

        shot = getattr(env, "last_shot", None)
        ttl = getattr(env, "last_shot_ttl", 0)

        if shot is not None and ttl > 0:
            def center(px: int, py: int) -> tuple[int, int]:
                return (px * cs + cs // 2, py * cs + cs // 2)

            for who, shot_data in shot.items():
                x0, y0, x1, y1, hit = shot_data
                p0 = center(x0, y0)
                p1 = center(x1, y1)
                if who == "player":
                    color = (60, 220, 120) if hit else (120, 220, 255)
                else:
                    color = (255, 180, 60) if hit else (220, 80, 80)
                pygame.draw.line(self._screen, color, p0, p1, 4)

        if text:
            font = pygame.font.SysFont(None, 20)
            surf = font.render(text, True, (230, 230, 230))
            self._screen.blit(surf, (6, 6))

        pygame.display.flip()
        self._clock.tick(self.fps)
