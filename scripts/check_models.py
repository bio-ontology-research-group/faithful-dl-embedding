#!/usr/bin/env python3
"""Numeric sanity checks for the constructions and obstructions in impossibility.tex.

These are *checks*, not proofs: the proofs are in impossibility.tex (and in lean/).

  1. Disc model of T_3 in R^2 (Prop. disks): axioms hold, A,B,C non-empty.
  2. Half-space model of T_k in R^{k-1} (Thm. dimension, <=): LP feasibility.
  3. Helly number 2 of boxes: random search never finds three pairwise
     intersecting boxes with empty triple intersection; random search over
     *discs* in R^2 does find such triples (so the box property is special).
  4. Axis-aligned cone (sign-set) model of T_3 in R^2 (Remark cones).
  5. Unions of intervals in R^1 realise T_k (Prop. nonconvex (1)).
"""
import itertools, math, sys
import numpy as np
from scipy.optimize import linprog

rng = np.random.default_rng(0)
ok = True
def report(name, cond):
    global ok
    ok &= bool(cond)
    print(f"[{'ok' if cond else 'FAIL'}] {name}")

# 1. discs -------------------------------------------------------------------
r = 1.1
v = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, math.sqrt(3)]])
xs = np.linspace(-1.5, 3.5, 1201); ys = np.linspace(-1.5, 3.5, 1201)
G = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
X = [np.linalg.norm(G - v[i], axis=1) <= r for i in range(3)]
A, B, C = X[0] & X[1], X[1] & X[2], X[0] & X[2]
report("discs: A,B,C non-empty", A.any() and B.any() and C.any())
report("discs: A&B, B&C, A&C empty on grid", not (A & B).any() and not (B & C).any() and not (A & C).any())
# analytic: circumradius 2/sqrt3 > r
report("discs: circumradius > r (analytic triple-emptiness)", 2 / math.sqrt(3) > r)

# 2. half-space model of T_k --------------------------------------------------
def halfspace_model_ok(k, eps=None):
    d = k - 1
    eps = eps if eps is not None else 1.0 / (k + 1)
    # lambda_j(x) = x_j for j<d, lambda_{d}(x) = 1 - sum x
    def constraints(J):  # A_ub x <= b_ub for {lambda_j <= eps : j in J}
        rows, rhs = [], []
        for j in J:
            if j < d:
                row = np.zeros(d); row[j] = 1; rows.append(row); rhs.append(eps)
            else:
                rows.append(-np.ones(d)); rhs.append(eps - 1)  # 1 - sum x <= eps
        return np.array(rows), np.array(rhs)
    def feasible(J):
        Aub, bub = constraints(J)
        res = linprog(np.zeros(d), A_ub=Aub, b_ub=bub, bounds=[(None, None)] * d, method="highs")
        return res.status == 0
    allk = list(range(k))
    return all(feasible([j for j in allk if j != i]) for i in allk) and not feasible(allk)
for k in range(2, 8):
    report(f"half-space model of T_{k} in R^{k-1}: (k-1)-wise feasible, k-wise infeasible", halfspace_model_ok(k))

# 3. Helly: boxes vs discs ----------------------------------------------------
def random_box(d):
    lo = rng.uniform(-1, 1, d); hi = lo + rng.uniform(0, 1.5, d); return lo, hi
def boxes_intersect(bs):
    lo = np.max([b[0] for b in bs], axis=0); hi = np.min([b[1] for b in bs], axis=0)
    return bool(np.all(lo <= hi))
found = False
for d in (1, 2, 3, 5):
    tries = 0
    while tries < 20000:
        bs = [random_box(d) for _ in range(3)]
        if all(boxes_intersect([bs[i], bs[j]]) for i, j in itertools.combinations(range(3), 2)):
            tries += 1
            if not boxes_intersect(bs):
                found = True
report("boxes: 80k random pairwise-intersecting triples, triple intersection always non-empty", not found)
def discs_triple_nonempty(cs, rs, n=4000):
    # exact: three discs have a common point iff some pairwise lens contains a point in the third;
    # approximate by sampling the lens boundary candidates and centres; use fine sampling
    pts = []
    for i, j in itertools.combinations(range(3), 2):
        dij = np.linalg.norm(cs[i] - cs[j])
        if dij > rs[i] + rs[j]: return False
        # candidate points: the two circle-circle intersection points and the centres
        a = (rs[i]**2 - rs[j]**2 + dij**2) / (2 * dij) if dij > 0 else 0
        h2 = rs[i]**2 - a**2
        if h2 >= 0 and dij > 0:
            p = cs[i] + a * (cs[j] - cs[i]) / dij
            perp = np.array([-(cs[j] - cs[i])[1], (cs[j] - cs[i])[0]]) / dij
            pts += [p + math.sqrt(h2) * perp, p - math.sqrt(h2) * perp]
    pts += list(cs)
    # a non-empty intersection of three discs (convex) contains one of: a centre inside the others,
    # or a pairwise circle intersection point inside the third disc
    return any(all(np.linalg.norm(p - cs[m]) <= rs[m] + 1e-9 for m in range(3)) for p in pts)
disc_counter = 0
for _ in range(20000):
    cs = rng.uniform(-1, 1, (3, 2)); rs = rng.uniform(0.2, 1.0, 3)
    pair = all(np.linalg.norm(cs[i] - cs[j]) <= rs[i] + rs[j] for i, j in itertools.combinations(range(3), 2))
    if pair and not discs_triple_nonempty(cs, rs):
        disc_counter += 1
report(f"discs: random search finds pairwise-intersecting triples with empty triple intersection ({disc_counter} found)", disc_counter > 0)

# 4. axis-aligned cones as sign sets ----------------------------------------
# cone <-> subset of half-axes {0+,0-,1+,1-}; intersection = set intersection; {0} = empty set
X1, X2, X3 = {"0+", "0-"}, {"0-", "1+"}, {"0+", "1+"}
Ac, Bc, Cc = X1 & X2, X2 & X3, X1 & X3
report("al-cones in R^2: A,B,C non-trivial and pairwise trivial",
       all(s for s in (Ac, Bc, Cc)) and not (Ac & Bc) and not (Bc & Cc) and not (Ac & Cc))

# 5. unions of intervals in R^1 ------------------------------------------------
def interval_union_model(k):
    # element i -> interval [i-1/4, i+1/4]; X_j = union over i != j
    pts = np.linspace(-0.5, k + 0.5, 20001)
    def region(S): return np.any([np.abs(pts - i) <= 0.25 for i in S], axis=0) if S else np.zeros_like(pts, bool)
    Xs = [region([i for i in range(k) if i != j]) for j in range(k)]
    As = [np.logical_and.reduce([Xs[j] for j in range(k) if j != i]) for i in range(k)]
    return all(a.any() for a in As) and not any((As[i] & As[j]).any() for i, j in itertools.combinations(range(k), 2))
for k in range(2, 7):
    report(f"unions of intervals in R^1 realise T_{k}", interval_union_model(k))

print("ALL OK" if ok else "SOME CHECKS FAILED"); sys.exit(0 if ok else 1)
