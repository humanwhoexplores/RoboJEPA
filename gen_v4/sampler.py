"""Goal-blind candidate pushes: sampled from the scene alone, never from a goal or an instruction.

This is the one change Wasiq asked for. In v3 most pushes were computed from the target point of
the scene's 4 instructions (gen_v3/candidates.py), so the candidate set carried the answer. Here
the sampler's only inputs are the scene (objects, their sizes and positions, distractor positions)
and a random stream of its own. It has no goal argument, and it never aims at zones or at spots
beside other objects, because those are exactly where goals live.

Per call (defaults are for the 64 pushes of a scene's start state):
    n_total - n_lines object pushes, split evenly over the named objects (remainder to random ones).
        For each object: directions evenly spaced around the circle with a random rotation and
        jitter inside each slot, distances stratified over [DIST_MIN, DIST_MAX] in random order,
        and a random contact offset across the object's face.
    n_lines random pusher lines anywhere on the table (as v3's random_line): some touch nothing,
        some hit several objects.
The list is shuffled, so a candidate's index carries no information.

The stored action is pusher-centric (x0, y0, theta, length), exactly as in v3. The sampler's own
object-centric numbers are kept under "sampler" for analysis only; they contain no goal information.
"""
import numpy as np

from gen_v3.candidates import clear_start, contact_radius, obstacles, random_line, to_action

N_TOTAL = 64
N_LINES = 8
N_STEP2 = 32  # pushes sampled from a two-push decision's middle state
N_STEP2_LINES = 4
DIST_MIN, DIST_MAX = 0.03, 0.25  # m the object is meant to travel
OFFSET_FRAC = 0.5  # contact offset uniform in +-OFFSET_FRAC * the object's contact radius


def per_object_counts(n_obj_pushes, n_objects, rng):
    base = np.full(n_objects, n_obj_pushes // n_objects)
    extra = rng.permutation(n_objects)[: n_obj_pushes - base.sum()]
    base[extra] += 1
    return base


def sample_pushes(scene, pos, dis_pos, rng, n_total=N_TOTAL, n_lines=N_LINES):
    """Goal-blind pushes for the objects at pos ((N, 3)), avoiding dis_pos ((D, 3)) when touching down.

    Returns a list of n_total candidate dicts in random order:
        {candidate_type: object_push | object_push_blocked | random_line,
         sampler: {object, theta, dist, offset} or None, x0, y0, theta, length}
    """
    objs = scene["objects"]
    obs = obstacles(scene, pos, dis_pos)
    counts = per_object_counts(n_total - n_lines, len(objs), rng)
    cands = []
    for k, n_k in enumerate(counts):
        if n_k == 0:
            continue
        r = contact_radius(objs[k])
        slot = 2 * np.pi / n_k
        phase = rng.uniform(0, 2 * np.pi)
        dist_bins = rng.permutation(n_k)
        for j in range(n_k):
            theta = float(np.angle(np.exp(1j * (phase + slot * (j + rng.uniform(-0.5, 0.5))))))
            dist = float(DIST_MIN + (dist_bins[j] + rng.uniform()) / n_k * (DIST_MAX - DIST_MIN))
            offset = float(rng.uniform(-OFFSET_FRAC, OFFSET_FRAC) * r)
            start, length = to_action(pos[k, :2], r, theta, dist, offset)
            start, extra = clear_start(start, theta, obs)
            ctype = "object_push"
            th = theta
            if start is None:  # boxed in by neighbours: fall back to a random line, as v3 does
                ctype = "object_push_blocked"
                start, th, length, extra = random_line(obs, rng)
            cands.append({"candidate_type": ctype,
                          "sampler": {"object": k, "theta": theta, "dist": dist, "offset": offset},
                          "x0": float(start[0]), "y0": float(start[1]), "theta": float(th),
                          "length": float(length + extra)})
    for _ in range(n_lines):
        start, theta, length, _ = random_line(obs, rng)
        cands.append({"candidate_type": "random_line", "sampler": None, "x0": float(start[0]),
                      "y0": float(start[1]), "theta": float(theta), "length": float(length)})
    return [cands[i] for i in rng.permutation(len(cands))]
