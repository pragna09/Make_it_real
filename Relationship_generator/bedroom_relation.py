import argparse, glob, itertools, json, math, sys
from collections import Counter, defaultdict
import numpy as np
 
HUMAN_CLEARANCE = 0.6          # m, accessible-space default (paper: use human body size)
VISIBILITY_CATEGORIES = {"tv", "television", "painting", "picture", "mirror", "poster"}
FRUSTUM_STEPS, FRUSTUM_LENGTH = 3, 2.0
SUPPORT_TOL = 0.08             # m, "resting on" tolerance in z
SHELL = {"door", "window"}
 
 
# ------------------------------------------------------------------ geometry
def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0
 
def circ_mean(deg):
    r = np.radians(np.asarray(deg, float))
    return math.degrees(math.atan2(np.sin(r).mean(), np.cos(r).mean()))
 
def circ_std(deg):
    r = np.radians(np.asarray(deg, float))
    R = math.hypot(np.cos(r).mean(), np.sin(r).mean())
    return min(180.0, math.degrees(math.sqrt(max(0.0, -2 * math.log(max(R, 1e-9))))))
 
def room_edges(w, d):
    """Rectangle (0,0)-(w,d), CCW, with inward normals."""
    pts = np.array([[0, 0], [w, 0], [w, d], [0, d]], float)
    edges = []
    for i in range(4):
        a, b = pts[i], pts[(i + 1) % 4]
        e = (b - a) / np.linalg.norm(b - a)
        edges.append((a, b, np.array([-e[1], e[0]])))      # left of a CCW edge = inward
    return edges
 
def nearest_wall(o, edges):
    """Wall closest to the object's box surface (center distance minus the box's half-extent
    along the wall normal). Using the box edge instead of the center keeps big objects like
    beds from flipping between two walls. Returns (center-to-wall distance, inward normal)."""
    f = front_vec(o["yaw"]); l = np.array([-f[1], f[0]])
    best = None
    for a, b, n in edges:
        dist = float(np.dot(o["xy"] - a, n))                       # center to wall line
        half = abs(np.dot(n, f)) * o["depth"] / 2 + abs(np.dot(n, l)) * o["width"] / 2
        gap = dist - half
        if best is None or gap < best[0]:
            best = (gap, dist, n)
    return best[1], best[2]
 
def front_vec(yaw):
    r = math.radians(yaw)
    return np.array([math.cos(r), math.sin(r)])
 
def to_local(pt, center, yaw):
    """world xy -> frame of an object: x along its front, y to its left."""
    r = math.radians(-yaw)
    c, s = math.cos(r), math.sin(r)
    d = np.asarray(pt, float) - np.asarray(center, float)
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
 
def inside_footprint(pt, o):
    loc = to_local(pt, o["xy"], o["yaw"])
    return abs(loc[0]) <= o["depth"] / 2 and abs(loc[1]) <= o["width"] / 2
 
 
# ------------------------------------------------------------------ loading
def make_obj(oid, cat, size, pos, rot_z):
    sx, sy, sz = size
    return {"id": oid, "category": cat, "width": sx, "depth": sy, "height": sz,
            "xy": np.array(pos[:2], float), "zc": pos[2] + sz / 2.0,   # base z -> center z
            "yaw": wrap180(float(rot_z) + 90.0)}                       # front is local +Y
 
def load_scene(path, include_shell):
    sc = json.load(open(path))
    r = sc["room"]
    W, D, H = float(r["width"]), float(r["depth"]), float(r.get("height", 0) or 0)
    objs = [make_obj(o["id"], o["category"], o["size"], o["position"], o.get("rotation_z", 0))
            for o in sc["objects"]]
    if include_shell:
        t = float(r.get("wall_thickness", 0.1))
        for kind, key in (("door", "doors"), ("window", "windows")):
            for i, s in enumerate(sc.get("shell", {}).get(key, [])):
                objs.append(make_obj(f"{kind}_{i+1:02d}", kind,
                                     [float(s["width"]), t, float(s["height"])],
                                     s["position"], s.get("rotation_z", 0)))
    edges = room_edges(W, D)
    for o in objs:
        d, n = nearest_wall(o, edges)
        f = front_vec(o["yaw"])
        o["d"] = d
        o["theta"] = wrap180(math.degrees(math.atan2(f[1], f[0]) - math.atan2(n[1], n[0])))
    for o in objs:
        o["parent"] = find_parent(o, objs)
        o["kind"] = classify(o, H)
    return sc, objs, H
 
 
def find_parent(o, objs):
    bottom = o["zc"] - o["height"] / 2
    best, best_top = None, -1e9
    for p in objs:
        if p is o or p["height"] < 0.05:      # rugs / floor covers don't "support" furniture
            continue
        top = p["zc"] + p["height"] / 2
        if abs(bottom - top) <= SUPPORT_TOL and inside_footprint(o["xy"], p) and top > best_top:
            best, best_top = p, top
    return best
 
def classify(o, H):
    if o["category"] in SHELL:
        return "shell"
    if o["parent"] is not None:
        return "second_tier"
    if o["height"] < 0.05 and o["zc"] - o["height"] / 2 < SUPPORT_TOL:
        return "floor_cover"                   # rugs: no wall prior, no accessible space
    bottom, top = o["zc"] - o["height"] / 2, o["zc"] + o["height"] / 2
    if H and top >= H - 0.15 and bottom > 1.0:
        return "ceiling"
    if bottom > SUPPORT_TOL + 0.3 and o["d"] < o["depth"] / 2 + 0.15:
        return "wall"
    return "floor"
 
 
# ------------------------------------------------------------------ statistics
def kmeans(X, k, iters=50, seed=0):
    rng = np.random.default_rng(seed)
    k = min(k, len(X))
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), axis=1)
        newC = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return lab
 
def spatial_clusters(objs, k):
    ds = np.array([o["d"] for o in objs])
    th = np.array([o["theta"] for o in objs])
    X = np.c_[ds, np.cos(np.radians(th)), np.sin(np.radians(th))]   # angle on a circle
    lab = kmeans(X, k) if k > 1 and len(X) > 1 else np.zeros(len(X), int)
    out = []
    for j in sorted(set(lab)):
        m = lab == j
        out.append({"d_mean": round(float(ds[m].mean()), 4), "d_std": round(float(ds[m].std()), 4),
                    "theta_mean": round(circ_mean(th[m]), 2), "theta_std": round(circ_std(th[m]), 2),
                    "weight": round(float(m.mean()), 3), "count": int(m.sum()),
                    "reliable": bool(m.sum() >= 5 and circ_std(th[m]) < 30)})
    return out
 
def accessible_space(objs):
    """Back is accessible only if some example has a real gap behind the object."""
    back = max(o["d"] - o["depth"] / 2 for o in objs) > 0.15
    s = {"front": HUMAN_CLEARANCE, "left": HUMAN_CLEARANCE, "right": HUMAN_CLEARANCE}
    if back:
        s["back"] = HUMAN_CLEARANCE
    return s
 
def viewing_frustum(width):
    step = FRUSTUM_LENGTH / FRUSTUM_STEPS
    return [{"center_dist_front": round(step * (i + .5), 3), "width": round(width * (1 + .3 * i), 3),
             "depth": round(step, 3)} for i in range(FRUSTUM_STEPS)]
 
def pair_samples(scenes, a, b, max_dist):
    out = []
    for objs in scenes:
        As = [o for o in objs if o["category"] == a]
        Bs = [o for o in objs if o["category"] == b]
        for x in As:
            cands = [y for y in Bs if y is not x]
            if not cands:
                continue
            y = min(cands, key=lambda o: np.linalg.norm(o["xy"] - x["xy"]))
            dist = float(np.linalg.norm(y["xy"] - x["xy"]))
            if dist > max_dist:
                continue
            loc = to_local(y["xy"], x["xy"], x["yaw"])
            out.append((dist, math.degrees(math.atan2(loc[1], loc[0])),
                        wrap180(y["yaw"] - x["yaw"]), loc))
    return out
 
 
# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("-o", "--out", default="relations.json")
    ap.add_argument("--k", type=int, default=1, help="k-means clusters for spatial priors")
    ap.add_argument("--pair", action="append", default=None,
                    help="catA:catB (repeatable). Default: auto-detect co-occurring pairs")
    ap.add_argument("--min-cooccur", type=float, default=0.2,
                    help="auto pairs: fraction of scenes both categories must appear in")
    ap.add_argument("--max-pair-dist", type=float, default=4.0)
    ap.add_argument("--include-shell", action="store_true", help="add doors/windows as objects")
    a = ap.parse_args()
 
    found = set()
    for p in a.scenes:
        m = glob.glob(p)
        if not m:
            print(f"warning: no file matches '{p}' (skipped)", file=sys.stderr)
        found.update(m)
    files = sorted(found)
    if not files:
        sys.exit("No scene files found.")
    if len(files) < 5:
        print(f"note: only {len(files)} scene(s); statistics (std, clusters) need more rooms "
              f"(the paper used 5 per scene type).", file=sys.stderr)
 
    scenes, raw = [], []
    for f in files:
        sc, objs, _ = load_scene(f, a.include_shell)
        scenes.append(objs); raw.append((sc, objs))
    allobjs = [o for s in scenes for o in s]
    cats = sorted({o["category"] for o in allobjs})
 
    res = {"meta": {"num_scenes": len(files), "files": files, "units": "meters / degrees",
                    "reference": "Yu et al. 2011, Make it Home, ACM TOG 30(4):86",
                    "angle_convention": "theta: angle between object front and inward normal of "
                                        "nearest wall; 0 = back to wall, facing into room"},
           "categories": {}, "hierarchy": {}, "pairwise": [], "symbolic_relations": []}
 
    # ---- categories: spatial prior, accessible space, frustum
    for c in cats:
        co = [o for o in allobjs if o["category"] == c]
        e = {"size_mean_xyz": [round(float(x), 3) for x in
                               np.mean([[o["width"], o["depth"], o["height"]] for o in co], axis=0)],
             "count_per_scene": round(len(co) / len(files), 2),
             "placement": dict(Counter(o["kind"] for o in co))}
        prior = [o for o in co if o["kind"] in ("floor", "wall")]
        if prior:
            e["spatial_prior"] = spatial_clusters(prior, a.k)
        floor = [o for o in co if o["kind"] == "floor"]
        if floor:
            e["accessible_space_m"] = accessible_space(floor)
        if c.lower() in VISIBILITY_CATEGORIES:
            e["viewing_frustum"] = viewing_frustum(e["size_mean_xyz"][0])
        res["categories"][c] = e
 
    # ---- hierarchy
    for c in cats:
        co = [o for o in allobjs if o["category"] == c]
        labels = [o["parent"]["category"] if o["parent"] is not None else o["kind"] for o in co]
        cnt = Counter(labels)
        h = {"parent_distribution": {k: round(v / len(co), 3) for k, v in cnt.items()},
             "tier": 2 if cnt.most_common(1)[0][0] not in ("floor", "wall", "ceiling", "shell", "floor_cover") else 1}
        offs = defaultdict(list)
        for o in co:
            if o["parent"] is not None:
                loc = to_local(o["xy"], o["parent"]["xy"], o["parent"]["yaw"])
                offs[o["parent"]["category"]].append([loc[0], loc[1], wrap180(o["yaw"] - o["parent"]["yaw"])])
        if offs:
            h["offset_on_parent"] = {p: {"local_xy_mean": [round(float(x), 3) for x in np.mean(v, axis=0)[:2]],
                                         "rel_yaw_mean": round(circ_mean([r[2] for r in v]), 2)}
                                     for p, v in offs.items()}
        res["hierarchy"][c] = h
 
    # ---- pairwise
    if a.pair:
        pairs = [tuple(s.split(":")) for s in a.pair]
    else:
        usable = [c for c in cats if all(o["kind"] not in ("ceiling", "shell", "second_tier", "floor_cover")
                                         for o in allobjs if o["category"] == c)]
        present = {c: sum(any(o["category"] == c for o in s) for s in scenes) / len(scenes) for c in usable}
        area = {c: np.mean([o["width"] * o["depth"] for o in allobjs if o["category"] == c]) for c in usable}
        pairs = []
        for x, y in itertools.combinations(usable, 2):
            if present[x] >= a.min_cooccur and present[y] >= a.min_cooccur:
                pairs.append((x, y) if area[x] >= area[y] else (y, x))   # larger object = anchor A
        if a.include_shell:
            for s in SHELL:
                if s in cats:
                    pairs += [(s, c) for c in usable if present[c] >= a.min_cooccur]
    for x, y in pairs:
        S = pair_samples(scenes, x, y, a.max_pair_dist)
        if not S:
            continue
        d, br, ry = [s[0] for s in S], [s[1] for s in S], [s[2] for s in S]
        locs = np.array([s[3] for s in S])
        brabs = [abs(b) for b in br]                     # fold left/right so both sides agree
        dm = float(np.mean(d))
        consistent = bool(len(S) >= 5 and np.std(d) < 0.25 * max(dm, 0.5) and np.std(brabs) < 30 and circ_std(ry) < 30)
        res["pairwise"].append({
            "a": x, "b": y, "count": len(S), "consistent": consistent,
            "dist_mean": round(dm, 4), "dist_std": round(float(np.std(d)), 4),
            "bearing_abs_mean": round(float(np.mean(brabs)), 2),
            "bearing_abs_std": round(float(np.std(brabs)), 2),
            "side_left_fraction": round(float((locs[:, 1] > 0).mean()), 3),
            "ahead_mean": round(float(locs[:, 0].mean()), 3),
            "lateral_abs_mean": round(float(np.abs(locs[:, 1]).mean()), 3),
            "rel_yaw_mean": round(circ_mean(ry), 2), "rel_yaw_std": round(circ_std(ry), 2),
            "bearing_mean": round(circ_mean(br), 2), "bearing_std": round(circ_std(br), 2),
            "note": "A's frame: ahead = along A's front, lateral = sideways. bearing_abs_* folds "
                    "left/right; side_left_fraction = share of B on A's left. Prefer bearing_abs_* "
                    "over bearing_mean (which cancels when B appears on both sides). "
                    "consistent = at least 5 samples and low spread in distance, bearing and relative yaw."})
 
    # ---- generator's own symbolic labels (optional bonus)
    sym = Counter()
    for sc, objs in raw:
        cat_of = {o["id"]: o["category"] for o in objs}
        for r in sc.get("relations", []):
            sym[(cat_of.get(r["subject"], r["subject"]), r["relation"],
                 cat_of.get(r["object"], r["object"]))] += 1
    res["symbolic_relations"] = [{"subject": s, "relation": r, "object": o, "count": n,
                                  "per_scene": round(n / len(files), 3)}
                                 for (s, r, o), n in sorted(sym.items())]
 
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"Wrote {a.out}: {len(res['categories'])} categories, {len(res['pairwise'])} pairwise "
          f"relations from {len(files)} scene(s)")
 
if __name__ == "__main__":
    main()