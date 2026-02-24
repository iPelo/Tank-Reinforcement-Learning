import time
import numpy as np

from src.env.tank_env import TankEnv
from src.env.render import PygameRenderer


def main(render: bool = True, episodes: int = 10) -> None:
    env = TankEnv(w=15, h=15, max_steps=200, seed=0, wall_density=0.12)
    rng = np.random.default_rng(0)

    renderer = PygameRenderer(cell_size=32, fps=30) if render else None

    for ep in range(episodes):
        _ = env.reset(phase=0)
        total_r = 0.0
        done = False

        while not done:
            action = int(rng.integers(0, 6))  # 0..5
            _, r, done, info = env.step(action)
            total_r += r

            if renderer is not None:
                si = info["info"]
                renderer.render(env, text=f"ep={ep} steps={si.steps} R={total_r:.2f} success={si.success}")

        si = info["info"]
        print(f"EP {ep}: steps={si.steps} total_reward={total_r:.2f} success={si.success}")

    if renderer is not None:
        time.sleep(0.25)
        renderer.close()


if __name__ == "__main__":
    main(render=True, episodes=10)