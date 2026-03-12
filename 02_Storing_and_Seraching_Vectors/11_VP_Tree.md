# VP-Trees (Vantage-Point Trees) — Notes

A VP-tree is a spatial index that works using **only a distance function**.

No coordinates. No dimensions. No dot products.
Just: "How far is A from B?"

---

## 1. Why VP-Trees Exist

**KD-trees assume:** coordinates exist, axes mean something.
**Ball trees assume:** you can represent regions with centers + radii.

**VP-trees assume only this:**
> The distance function is a valid metric (non-negative, symmetric, triangle inequality holds).

This means VP-trees work for:
- Strings (edit distance)
- Graphs (shortest-path distance)
- Audio fingerprints
- Shapes
- Anything with a metric

That generality is their power.

---

## 2. The Core Idea (One Sentence)

Pick a point, measure distances to all others, split into **near** and **far** based on a radius, recurse.

That's the entire tree.

---

## 3. What a VP-Tree Node Stores

- **Vantage point** `vp` — one actual data point
- **Threshold radius** `r`
- **Two children:**
  - Inner subtree: points with `dist(vp, x) < r`
  - Outer subtree: points with `dist(vp, x) >= r`

```
            vp
           /  \
     dist < r  dist >= r
      inner     outer
```

Every point goes into exactly one subtree.

---

## 4. How a VP-Tree Is Built

Given a set of points `S`:

### Step 1: Choose a vantage point

Pick one point from `S` as `vp`.
- Most common: random point
- Fancier: farthest-from-mean heuristic

> VP-trees don't need a "good" vp to be correct — only to be efficient.

### Step 2: Compute distances

For every other point `x` in `S`:
```
d(x) = dist(vp, x)
```

### Step 3: Choose radius `r`

Set `r` = **median** of all distances.

Why median? Balances the tree — inner and outer get ~N/2 points each.

### Step 4: Partition

- Inner: points with `d(x) < r`
- Outer: points with `d(x) >= r`

Remove `vp` itself from children.

### Step 5: Recurse

Repeat on inner and outer sets. Stop when node has 1 point or <= bucket size.

---

## 5. Concrete Build Example

Points on a number line (using absolute difference as distance):

```
Points: {1, 3, 4, 7, 8, 10}
```

**Choose vp = 7.** Compute distances:

```
|7-1| = 6
|7-3| = 4
|7-4| = 3
|7-8| = 1
|7-10| = 3
```

Sorted distances: `1, 3, 3, 4, 6` → median = 3 → `r = 3`

**Partition:**
- Inner (dist < 3): {8} (dist=1)
- Outer (dist >= 3): {1, 3, 4, 10} (dists: 6, 4, 3, 3)

```
        vp=7 [r=3]
       /          \
    {8}    {1, 3, 4, 10}
   inner       outer
```

Recurse on both sides.

---

## 6. Nearest-Neighbor Search

Given query `q`, keep `best_dist` (initially infinity) and `best_point`.

At a node with `(vp, r)`:

### Step A: Check the vantage point

```
d_q = dist(q, vp)
```
If `d_q < best_dist` → update best.

### Step B: Visit the more promising subtree first

```
if d_q < r:
    visit inner first    # query is closer to inner region
else:
    visit outer first    # query is closer to outer region
```

Visiting the promising side first tightens `best_dist` before evaluating the other side — same principle as ball trees.

### Step C: Prune the other subtree

After searching the first subtree, decide if the other could contain a closer point:

**Can inner subtree have a closer point?**
```
if d_q - best_dist < r:    # yes, must check inner
```

**Can outer subtree have a closer point?**
```
if d_q + best_dist >= r:   # yes, must check outer
```

If the condition fails → **prune that entire subtree**.

---

## 7. Why the Pruning Rules Work (Intuition)

Think in distance shells around `vp`:

```
      |<--- inner --->|<--- outer --->
      0               r               ...
                      ^
                  threshold
```

- All inner points are within distance `r` of `vp`
- All outer points are beyond distance `r` from `vp`

The **triangle inequality** tells you: the distance from `q` to any point `x` is bounded by `|dist(q,vp) - dist(vp,x)|` and `dist(q,vp) + dist(vp,x)`.

So if `d_q` (query's distance to vp) is far from `r`, and `best_dist` is small, then one side of the partition is geometrically unreachable — no point there can beat your current best.

**Example:** If `d_q = 2`, `r = 10`, and `best_dist = 3`:
- Outer points are all at distance >= 10 from vp
- Closest outer point to q is at least `10 - 2 = 8` away
- 8 > 3 (best_dist) → prune outer entirely

---

## 8. Why VP-Trees Are Elegant

They:
- Work with **any metric** — not just Euclidean
- Require **no vector arithmetic** — just a distance function
- Don't care about axes or coordinates
- Don't store bounding boxes or balls with explicit centers

They are *pure metric geometry*.

---

## 9. Why VP-Trees Still Fail in High Dimensions

Same enemy. Different battlefield.

In high dimensions:
- Distances concentrate — `d_q` is often close to `r`
- Pruning inequalities rarely trigger
- Both subtrees must be searched
- Tree traversal degenerates toward brute force

**Rule of thumb:**
- VP-trees work well for: string edit distance, graph distances, low-moderate dimensional metric spaces
- They struggle for: high-dimensional dense embeddings (384+ dims)

---

## 10. KD-Tree vs Ball Tree vs VP-Tree

| Property | KD-Tree | Ball Tree | VP-Tree |
|---|---|---|---|
| Uses coordinates | Yes | Yes | **No** |
| Uses axes | Yes | No | No |
| Requires only metric | No | Partially | **Yes** |
| Region shape | Boxes | Balls | Distance shells |
| High-dim behavior | Poor (~20-30d) | Slightly better (~50-100d) | Poor |
| Works for strings/graphs | No | Rarely | **Yes** |

---

## 11. Why VP-Trees Matter Even If You Won't Use Them

VP-trees teach a fundamental lesson:

> **ANN is not about vectors — it's about distance structure.**

This perspective is crucial when you later compare trees, IVF, graphs, and PQ. They all try to exploit distance geometry differently. VP-trees make that idea the most explicit.

---

## Key Takeaways

1. **VP-tree = partition by distance to a reference point**, split into near/far at median distance
2. **Only needs a metric** — no coordinates, no axes, no vector math
3. **Pruning uses triangle inequality** — if the best case for a subtree is already worse than `best_dist`, skip it
4. **Visit order matters** — check the promising subtree first to tighten bounds early
5. **Still falls to curse of dimensionality** — distance concentration kills pruning in high dims
6. **Foundational insight:** ANN is about distance structure, not vector arithmetic

---