"""Generate push trajectories.

Each scene: one start image, then M candidate push sequences, each executed from the same
saved simulator state. Stores the image after every push and the resulting region.

    python -m robojepa.collect --out data --workers 5
"""
import argparse
import multiprocessing as mp
import os
import time

import numpy as np

# split -> (num scenes, candidates per scene, candidate mode, spawn-x range, seed offset)
SPLITS = {
    "train": (400, 16, "random", (-0.08, 0.04), 0),
    "val": (50, 16, "random", (-0.08, 0.04), 100_000),
    "test_id": (100, 48, "eval", (-0.08, 0.04), 200_000),
    "test_ood": (60, 48, "eval", (0.04, 0.08), 300_000),  # held-out cube positions
}
MAX_PUSH = 2
CONTACT_BACKOFF = 0.055  # finger starts this far behind the estimated cube center
DETECTOR_NOISE = 0.01  # std (m) of the simulated cube-position estimate used to place candidates


def _unit(theta):
    return np.array([np.cos(theta), np.sin(theta)])


def _push(rng, center, theta, length, lateral_std):
    d = _unit(theta)
    perp = np.array([-d[1], d[0]])
    s = center - CONTACT_BACKOFF * d + perp * rng.normal(0, lateral_std)
    return np.concatenate([s, s + length * d])


def _sample_theta(rng):
    # Bias toward the two region directions (+-y) so that successes are not too rare.
    if rng.random() < 0.7:
        return rng.choice([np.pi / 2, -np.pi / 2]) + rng.normal(0, np.deg2rad(25))
    return rng.uniform(-np.pi, np.pi)


def random_candidates(rng, cube_est, m):
    seqs = np.zeros((m, MAX_PUSH, 4), np.float32)
    n_push = np.where(rng.random(m) < 0.3, 2, 1)
    for i in range(m):
        center = cube_est
        for k in range(n_push[i]):
            p = _push(rng, center, _sample_theta(rng), rng.uniform(0.08, 0.30), 0.015)
            seqs[i, k] = p
            # Heuristic guess of where the cube ends up: just ahead of the finger.
            d = (p[2:] - p[:2]) / (np.linalg.norm(p[2:] - p[:2]) + 1e-9)
            center = p[2:] + d * CONTACT_BACKOFF
    return seqs, n_push


def eval_candidates(rng, cube_est, m):
    """Fixed structured set: 8 directions x 3 lengths single pushes, then random 2-push sequences."""
    singles = [_push(rng, cube_est, th, L, 0.0)
               for th in np.arange(8) * np.pi / 4 for L in (0.10, 0.18, 0.26)]
    seqs = np.zeros((m, MAX_PUSH, 4), np.float32)
    n_push = np.ones(m, int)
    seqs[: len(singles), 0] = singles
    rest, rest_n = random_candidates(rng, cube_est, m - len(singles))
    rest_n[:] = 2
    for i in range(len(rest)):  # force two pushes for the remainder
        if not rest[i, 1].any():
            p = rest[i, 0]
            d = (p[2:] - p[:2]) / (np.linalg.norm(p[2:] - p[:2]) + 1e-9)
            rest[i, 1] = _push(rng, p[2:] + d * CONTACT_BACKOFF, _sample_theta(rng), rng.uniform(0.08, 0.30), 0.015)
    seqs[len(singles):] = rest
    n_push[len(singles):] = rest_n
    return seqs, n_push


def run_scene(args):
    split, idx, out_dir = args
    path = os.path.join(out_dir, split, f"scene_{idx:05d}.npz")
    if os.path.exists(path):
        return path
    import robojepa.env as E

    n_scenes, m, mode, spawn_x, seed_off = SPLITS[split]
    E.SPAWN_X = spawn_x
    env = _get_env()
    rng = np.random.default_rng(seed_off + idx)
    scene = env.reset_scene(rng)
    start_img = env.render_image()
    state = env.get_state()
    cube_est = scene["cube_xy"] + rng.normal(0, DETECTOR_NOISE, 2)
    gen = random_candidates if mode == "random" else eval_candidates
    seqs, n_push = gen(rng, cube_est, m)

    H = env.img_size
    imgs = np.zeros((m, MAX_PUSH, H, H, 3), np.uint8)
    region = np.full(m, -1, np.int8)  # -1 none, 0 left, 1 right
    final_xy = np.zeros((m, 2), np.float32)
    for i in range(m):
        env.set_state(state)
        for k in range(n_push[i]):
            # Every push ends with the arm retracted, so all stored frames are comparable decision points.
            env.execute_push(seqs[i, k])
            stayed = env.settle()
            imgs[i, k] = env.render_image()
        region[i] = {None: -1, "left": 0, "right": 1}[stayed]
        final_xy[i] = env.cube_pos[:2] - E.TABLE_TOP[:2]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + ".tmp.npz"
    np.savez_compressed(
        tmp, start_img=start_img, imgs=imgs, seqs=seqs, n_push=n_push.astype(np.int8), region=region,
        final_xy=final_xy, start_xy=scene["cube_xy"].astype(np.float32), cube_est=cube_est.astype(np.float32),
        colors=np.array(scene["colors"]))
    os.replace(tmp, path)
    return path


_ENV = None


def _get_env():
    global _ENV
    if _ENV is None:
        from robojepa.env import PushEnv
        _ENV = PushEnv()
    return _ENV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--limit", type=int, default=None, help="max scenes per split (for smoke tests)")
    a = ap.parse_args()
    jobs = [(s, i, a.out) for s in a.splits for i in range(min(SPLITS[s][0], a.limit or 10**9))]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for n, _ in enumerate(pool.imap_unordered(run_scene, jobs), 1):
            if n % 10 == 0 or n == len(jobs):
                el = time.time() - t0
                print(f"{n}/{len(jobs)} scenes  {el / 60:.1f} min  eta {el / n * (len(jobs) - n) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
