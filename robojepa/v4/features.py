"""Cache frozen features for the team's v4 dataset (gen_v4, branch generator-v4).

Per scene: the start frame and the settled final frame of each of the 64 goal-blind pushes, encoded
with DINOv2-small and stored as int8 with a per-token scale (half the disk of float16). Every goal of
every scene gets its instruction embedded with MiniLM and its 64 labels. Two-push (step 2) data is
not used yet.

    python -m robojepa.v4.features --data data/ext_v4/data/v4 --code data/ext_v4/code --out features_v4

Splits: v4's val scenes are cut in two by scene order, so we get an in-distribution test set
("test_id") next to v4's test split (held-out templates and shape words; half of it also held-out
cameras). Splits written: train, val, test_id, test.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from robojepa.features import VisionEncoder, TEXT_MODEL

N_CAND = 64


def load_scene(data, sid):
    """Start image, final images (64), labels, goals and per-push contact info for one scene."""
    sc = json.load(open(f"{data}/scenes/{sid}.json"))
    lab = np.load(f"{data}/scenes/{sid}_labels.npz")
    start = np.load(f"{data}/scenes/{sid}_initial.npz")["initial"]
    finals, touched, actions = [], [], []
    for tid in lab["traj_ids"]:
        t = np.load(f"{data}/trajs/{tid}.npz")
        finals.append(t["keyframes"][list(t["keyframe_names"]).index("final")])
        touched.append(t["contact"].any(0))
        c = json.loads(str(t["meta"]))["candidate"]
        actions.append([c["x0"], c["y0"], c["theta"], c["length"]])
    return dict(sc=sc, goal_ids=lab["goal_ids"], labels=lab["default"], start=start, finals=np.stack(finals),
                touched=np.stack(touched), actions=np.array(actions, np.float32))


def rule_pick(goal, sc, touched, actions):
    """v4's no-physics reference rule (gen_v4/validate.py): among pushes that touched the named object,
    the one whose direction best matches the line from the object to the goal's target point.
    Uses privileged state (positions, contacts). Returns a candidate index, or -1 (none touched it)."""
    from gen_v3.tasks import target_point
    pos = np.asarray(sc["init_pos"])
    v = target_point(goal, sc["spec"], pos) - pos[goal["a"], :2]
    tm = touched[:, goal["a"]]
    if not tm.any():
        return -1
    ix = np.flatnonzero(tm)
    return int(ix[np.argmax(np.cos(actions[ix, 2] - np.arctan2(v[1], v[0])))])


def quantize(z):
    """(B,T,D) float -> int8 codes and a float16 per-token scale."""
    s = z.abs().amax(-1, keepdim=True).clamp_min(1e-6) / 127
    return (z / s).round().clamp(-127, 127).to(torch.int8).cpu().numpy(), s.squeeze(-1).half().cpu().numpy()


def encode_split(enc, data, sids, out, split, texts):
    n = len(sids)
    codes = np.lib.format.open_memmap(f"{out}/{split}_z.npy", mode="w+", dtype=np.int8, shape=(n * (N_CAND + 1), 256, 384))
    scales = np.zeros((n * (N_CAND + 1), 256), np.float16)
    g_scene, g_text, g_type, g_id, g_lab, g_rule, g_eval, g_source, g_straight = ([] for _ in range(9))
    actions = np.zeros((n, N_CAND, 4), np.float32)
    with ThreadPoolExecutor(8) as pool:
        for i, d in enumerate(pool.map(lambda s: load_scene(data, s), sids)):
            sc = d["sc"]
            rows = slice(i * (N_CAND + 1), (i + 1) * (N_CAND + 1))  # row 0: start frame, rows 1..64: finals
            q, s = quantize(enc(np.concatenate([d["start"][None], d["finals"]])).float())
            codes[rows], scales[rows] = q, s
            actions[i] = d["actions"]
            evals = {e["goal_id"]: e["source"] for e in sc["eval_goals"]}
            col = {g: k for k, g in enumerate(d["goal_ids"])}
            for g in sc["goals"]:
                gid = g["goal_id"]
                g_scene.append(i), g_type.append(g["type"]), g_id.append(gid)
                g_text.append(texts.setdefault(sc["instructions"][gid], len(texts)))
                g_lab.append(d["labels"][:, col[gid]])
                g_rule.append(rule_pick(g, sc, d["touched"], d["actions"]))
                g_eval.append(gid in evals), g_source.append(evals.get(gid, "")), g_straight.append(g["straight_path"])
            if i % 50 == 0:
                print(f"{split}: {i}/{n} scenes", flush=True)
    codes.flush()
    np.save(f"{out}/{split}_scale.npy", scales)
    cams = [json.load(open(f"{data}/scenes/{s}.json"))["spec"]["camera"] for s in sids]
    np.savez(f"{out}/{split}_meta.npz", scene_ids=np.array(sids), camera=np.array(cams), actions=actions,
             g_scene=np.array(g_scene), g_text=np.array(g_text), g_type=np.array(g_type), g_id=np.array(g_id),
             g_lab=np.stack(g_lab).astype(np.int8), g_rule=np.array(g_rule), g_eval=np.array(g_eval),
             g_source=np.array(g_source), g_straight=np.array(g_straight))
    print(f"{split}: {n} scenes, {len(g_id)} goals ({sum(g_eval)} eval decisions)")


@torch.no_grad()
def encode_texts(texts, out, device):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
    model = AutoModel.from_pretrained(TEXT_MODEL).to(device).eval()
    order = sorted(texts, key=texts.get)
    embs = []
    for i in range(0, len(order), 1024):
        b = tok(order[i:i + 1024], padding=True, return_tensors="pt").to(device)
        h = model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        embs.append(torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).cpu().numpy())
    np.savez(f"{out}/text.npz", emb=np.concatenate(embs).astype(np.float32), text=np.array(order))
    print(f"text: {len(order)} distinct instructions")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ext_v4/data/v4")
    ap.add_argument("--code", default="data/ext_v4/code", help="folder containing gen_v3/ (for the reference rule)")
    ap.add_argument("--out", default="features_v4")
    ap.add_argument("--limit", type=int, default=None, help="first N scenes per split (smoke test)")
    a = ap.parse_args()
    sys.path.insert(0, a.code)
    os.makedirs(a.out, exist_ok=True)
    sids = sorted(f[:-5] for f in os.listdir(f"{a.data}/scenes") if f.endswith(".json") and "rows" not in f)
    by_split = {}
    for s in sids:
        by_split.setdefault(json.load(open(f"{a.data}/scenes/{s}.json"))["spec"]["split"], []).append(s)
    val = by_split.pop("val")
    by_split["val"], by_split["test_id"] = val[: len(val) // 2], val[len(val) // 2:]
    device = "cuda"
    enc, texts = VisionEncoder(device), {}
    for split in ("train", "val", "test_id", "test"):
        encode_split(enc, a.data, by_split[split][: a.limit], a.out, split, texts)
    encode_texts(texts, a.out, device)


if __name__ == "__main__":
    main()
