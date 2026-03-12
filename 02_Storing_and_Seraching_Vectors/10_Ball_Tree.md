# Ball Trees — Notes

A ball tree partitions space using **balls (spheres)** instead of axis-aligned cuts.

**Why it exists:**
- In many datasets, axes are meaningless (especially after embeddings/transforms)
- A ball (distance-based region) can match the shape of clusters better than rectangles
- Works naturally with any metric distance, not just Euclidean

---

## 1. What a Ball Tree Node Represents

Each node represents a **region of space** shaped like a ball:

- **Center** `c` (a vector)
- **Radius** `r`
- A set of points inside that ball

Meaning:
> Every point stored in this node is within distance `r` of `c`.

A node has two children (left ball, right ball). The parent ball contains both child balls.

---

## 2. How a Ball Tree Is Built

Goal: recursively split points into two groups so each fits into a tighter ball.

### Step A: Pick two "far apart" points as anchors

You want two points near opposite sides of the cluster.

**Heuristic:**
1. Pick a random point `p`
2. Find the farthest point from `p` → call it `a`
3. Find the farthest point from `a` → call it `b`

Now `a` and `b` are "extreme" points — cheap approximation of the widest spread.

### Step B: Split points by which anchor they're closer to

For each point `x`:
- If `dist(x, a) < dist(x, b)` → group A
- Else → group B

This is essentially a single-step 2-means clustering (no iteration needed).

### Step C: Recurse

Build child nodes for each group. Stop when leaf size is small enough.

### Step D: Define the ball for each node

For a node containing points S:
1. **Center** `c` = mean of points (centroid) or a representative point (medoid)
2. **Radius** `r = max(dist(c, x))` over all x in S

That's your ball — the tightest sphere that contains all points in the node.

---

## 3. The Pruning Rule (The Entire Point)

For any node with ball `(c, r)`, the **minimum possible distance** from query `q` to any point inside that ball is:

```
lower_bound = max(0, dist(q, c) - r)
```

**Why:**
- `dist(q, c)` is distance from query to ball center
- The closest point inside the ball could be up to `r` closer (sitting on the edge toward q)
- So best case is `dist(q, c) - r`
- If query is inside the ball, minimum is 0

```
         r
    |---------|
    •----c----•         q
    ^ball^          ^query^

    dist(q, c) = 10, r = 3
    lower_bound = 10 - 3 = 7
```

**Pruning decision:**
```
if lower_bound > best_dist:
    # entire ball can't contain a closer point → PRUNE
```

This is ball tree's equivalent of KD-tree's plane pruning, but more natural — you're asking "how close could anything in this sphere possibly be?"

---

## 4. The Search Procedure

Given query `q`, find nearest point:

### Step A: Start at root
- `best_point = null`
- `best_dist = infinity`

### Step B: At each node, choose which child to visit first

Compute lower bounds for both children:
```
lb_left  = max(0, dist(q, c_left)  - r_left)
lb_right = max(0, dist(q, c_right) - r_right)
```

**Visit the child with the smaller lower bound first.**

### Step C: Recurse and prune

1. If node is a **leaf**: check all points, update `best_dist`
2. Else:
   - Compute bounds for both children
   - Visit the more promising child first
   - Update `best_dist` as you find closer points
   - **Only visit the other child if its lower bound <= best_dist**

**Why visit order matters:** A better `best_dist` early = more pruning later. Visiting the promising side first tightens your bound before you evaluate the other side.

---

## 5. Why Ball Trees Can Beat KD-Trees

KD-tree pruning checks distance to a **plane** (one dimension at a time).
Ball tree pruning checks distance to a **region** (all dimensions at once).

Ball trees are better when:
- Clusters are "round-ish" rather than axis-aligned
- Data isn't aligned to coordinate axes
- Individual dimensions don't have meaningful interpretation
- You're using a non-Euclidean metric (ball trees only need a distance function)

---

## 6. Ball Trees Still Break in High Dimensions

Ball trees are still subject to the curse of dimensionality:

- Balls overlap heavily in high dimensions
- `dist(q, c)` values bunch together (distance concentration)
- Radii become large relative to inter-point distances
- Lower bounds become weak → pruning fails
- You end up visiting most nodes anyway

**Rule of thumb:**
- Ball trees push a bit farther than KD-trees (~50–100 dims depending on data)
- But still degrade toward brute force well before embedding dimensions
- For 768-d embeddings, not practical

---

## 7. When Ball Trees Are Used Today

- **Moderate dimensions** (10–100-ish)
- **Non-Euclidean metrics** that are still metric (triangle inequality holds)
- **Exact neighbors** required
- **Classical ML** pipelines (kNN classification, clustering acceleration)
- **Scientific data**, robotics features

Not typically used for large-scale semantic embedding search.

---

## 8. KD-Tree vs Ball Tree

| Feature | KD-Tree | Ball Tree |
|---|---|---|
| Partition shape | Boxes (axis-aligned) | Spheres (distance-based) |
| Best when | Axes are meaningful | Axes are meaningless / clusters are round |
| Pruning bound | Distance to split plane | Distance to ball surface |
| Metric requirement | Euclidean-like | Any metric distance |
| High-dim behavior | Breaks ~20-30 dims | Breaks ~50-100 dims, still breaks |

---

## Checkpoint Question

A node has center distance `dist(q, c) = 10` and radius `r = 3`.

What is the minimum possible distance from `q` to any point inside that ball?

```
lower_bound = max(0, dist(q, c) - r) = max(0, 10 - 3) = 7
```

---

## Key Takeaways

1. **Ball tree = recursive distance-based splits** using spheres, not axis cuts
2. **Pruning = lower bound on distance to ball** — if `dist(q,c) - r > best_dist`, skip the whole subtree
3. **Visit order matters** — check the more promising child first to tighten `best_dist` early
4. **Better than KD-trees** when data is clustered and axes are meaningless
5. **Still breaks in high dimensions** — curse of dimensionality applies to all spatial partitioning
6. **This is distance-based partitioning** (from section 8B of file 08) made concrete

---