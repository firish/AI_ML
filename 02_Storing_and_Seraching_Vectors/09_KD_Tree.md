# KD-Trees (K-Dimensional Trees) — Notes

## 1. The Problem

**Given:**
- N points (vectors), each of dimension d
- Query: "what is the nearest point to this query vector?"

**Brute force:** Compute distance to all N, take minimum. Too slow at scale.

**KD-tree idea:** Split space into regions, quickly ignore whole regions that can't contain the answer.

Think "Google Maps grid cuts" but in many dimensions.

---

## 2. The Data Structure

Each node in a KD-tree stores:

- **One point** (or a small bucket of points at leaves)
- **A split rule:**
  - `axis`: which dimension to split on (0..d-1)
  - `s`: split value (usually the point's coordinate along that axis)
- **Two children:**
  - Left: points with `coordinate < s` along axis
  - Right: points with `coordinate >= s` along axis

Every node is basically saying:
> "I split the world into left vs right using this one dimension."

---

## 3. How a KD-Tree Is Built

### Standard Build Recipe (Balanced KD-Tree)

1. **Choose a split axis:**
   - Simple: cycle through dimensions (x, y, x, y, ...)
   - Better: pick the axis with **highest variance** (wider spread = better separation)

2. **Choose a split value:**
   - Use the **median** point along that axis
   - This keeps the tree balanced (equal points on each side)

3. **Put the median point at this node**

4. **Recursively build:**
   - Left subtree with points left of median
   - Right subtree with points right of median

5. **Stop** when leaf nodes have very few points

### Why Median Matters

- Left and right get ~N/2 points each
- Tree depth is ~log2(N)
- Search stays efficient

---

## 4. Concrete Build Example (2D)

**Points:**
```
A(2,3)  B(5,4)  C(9,6)  D(4,7)  E(8,1)  F(7,2)
```

**Level 0 — split on x:**
```
Sort by x: A(2,3), D(4,7), B(5,4), F(7,2), E(8,1), C(9,6)
Median = F(7,2)
```
- Root = F(7,2), split axis = x, split value = 7
- Left (x < 7): A, D, B
- Right (x >= 7): E, C

**Level 1 — split on y:**

Left subtree {A(2,3), B(5,4), D(4,7)}:
```
Sort by y: A(3), B(4), D(7)
Median = B(5,4)
```
- Node = B(5,4), split axis = y, split value = 4
- Left-left (y < 4): A(2,3) → leaf
- Left-right (y >= 4): D(4,7) → leaf

Right subtree {E(8,1), C(9,6)}:
```
Sort by y: E(1), C(6)
Median = C(9,6) or E(8,1) — pick one
```

**Result:** A balanced binary tree, depth ~log2(6) ≈ 3.

```
            F(7,2) [x=7]
           /              \
     B(5,4) [y=4]      E(8,1) [y=1]
      /       \              \
   A(2,3)   D(4,7)        C(9,6)
```

---

## 5. How Nearest Neighbor Search Works

Given query point `q`:

### Step A: Descend (like binary search)

At each node:
- Compare query's coordinate on the split axis with the split value
- Go left or right accordingly
- Reach a leaf — this is the region that "contains" the query

### Step B: Track best-so-far

At the leaf:
- Compute distance to the leaf's point(s)
- Store as current best candidate and `best_dist`

### Step C: Backtrack and prune

Walk back up the tree. At each parent node, ask:

> Could the other side of the split contain a closer point?

- **No** → prune that entire branch
- **Yes** → explore that side too, update best if needed

---

## 6. The Pruning Rule (The Key to Speed)

At a node splitting on axis `a` with split value `s`:

```
plane_dist = |q[a] - s|     # distance from query to the splitting plane
```

```
if plane_dist > best_dist:
    # other side can't possibly have a closer point
    # PRUNE — skip entire subtree
```

**Why this works:** Any point on the other side must be at least `plane_dist` away from the query (just to cross the splitting plane). If that's already farther than your current best, no point checking.

**This pruning is where KD-trees win.** In low dimensions, most branches get pruned, so you visit a small fraction of the tree.

---

## 7. Why KD-Trees Work Well in Low Dimensions

In 2D/3D:
- Regions are nicely separated
- `plane_dist > best_dist` is often true → heavy pruning
- You visit a small part of the tree

Search time is close to **O(log N)** in practice.

---

## 8. Why KD-Trees Break in High Dimensions

In high dimensions (100–768 dims):

- The query is close to the splitting plane in **some** dimension almost always
- `plane_dist` is often small
- `plane_dist > best_dist` is **rarely true**
- You can't prune → you visit most nodes
- Performance degrades to **O(N)** — effectively brute force

This is the **curse of dimensionality** in action.

**Rule of thumb:**
- KD-trees work well for **d <= 20–30**
- Beyond that, pruning stops being effective
- Modern embeddings (384/768/1536 dims) are way beyond KD-tree territory

---

## 9. When KD-Trees Are Still Used Today

KD-trees remain useful when:
- **Dimension is low** — geospatial (2D/3D), robotics, small feature vectors
- **You need exact nearest neighbors** — no approximation acceptable
- **Data is static** — build once, query many times

**Not the tool for** modern LLM embeddings or high-dimensional vector search.

---

## Key Takeaways

1. **KD-tree = recursive axis-aligned median splits** forming a balanced binary tree
2. **Search = descend to leaf, then backtrack** with pruning
3. **Pruning depends on distance to splitting plane** — skip a branch if crossing the plane is already too far
4. **Low dimensions: fast** (heavy pruning, ~O(log N))
5. **High dimensions: collapses** (no pruning, ~O(N), curse of dimensionality)
6. **This is axis-aligned partitioning** (from section 8A of the previous file) made concrete
