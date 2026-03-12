# Space Partitioning: The Idea Before the Index — Notes

## 1. What Does "Partitioning Space" Mean?

You have a space full of vectors. Partitioning means:

> Divide the space into regions so each point belongs to exactly one region.

**Analogies:**
- A map divided into countries
- A city divided into neighborhoods
- A warehouse divided into aisles

Each region:
- Covers a subset of the space
- Contains some points
- Has boundaries

The goal is not beauty. **The goal is faster search.**

---

## 2. Why Partition Space at All?

Without partitioning: check every point (brute force).

With partitioning:
1. Figure out which region the query falls into
2. Search only that region
3. Skip everything else

Replaces "check everything" with "check a small subset."

That's the entire motivation.

---

## 3. Starting Simple: 1D

Points on a number line:

```
|----|----|----|----|
```

Cut at a value `c`:

```
<--- left --->|<--- right --->
         c
```

If query falls left → search only left points, ignore right.

Simple. Powerful. **Exact** in 1D.

---

## 4. Extending to 2D

Now points are on a plane. Partition options:
- **Vertical cut:** x < c vs x >= c
- **Horizontal cut:** y < c vs y >= c
- **Any arbitrary line**

```
| Region A | Region B |
|----------|----------|
| Region C | Region D |
```

If query is in Region B → search Region B first, maybe check nearby regions if needed.

**Key idea:** Distance-aware pruning becomes possible — you can skip entire regions that are geometrically far from the query.

---

## 5. What Makes a "Good" Partition?

Three competing goals:

**(1) Regions should be small**
- Fewer points per region → faster local search

**(2) Regions should be simple**
- Easy to test "which region am I in?"
- Cheap boundary checks

**(3) Nearby points should land in the same region**
- Otherwise nearest neighbors cross partition boundaries
- Forces you to search multiple regions anyway

**These goals conflict.** You can't maximize all three. Every partitioning scheme is a tradeoff between them.

---

## 6. The Boundary Problem

Partition boundaries are artificial. Geometry is continuous.

So this happens constantly:

```
        |
   q •  |  • true nearest neighbor
        |
    Region A | Region B
```

Query is near a boundary. True nearest neighbor is just across the border.

**Any partition-based system must:**
- Either search multiple regions (slower but correct)
- Or risk missing true neighbors (faster but approximate)

This tradeoff is **unavoidable** — it's not a bug in any specific algorithm, it's fundamental to the approach.

---

## 7. Why Trees Naturally Emerge

If you partition recursively, you get a tree:

```
         [Entire Space]
          /          \
    [Left half]   [Right half]
     /     \        /     \
   [LL]   [LR]   [RL]    [RR]
```

- **Root** = entire space
- **Internal nodes** = split decisions
- **Leaves** = small regions with few points

**Search becomes:**
1. Start at root
2. Descend to the leaf region containing the query
3. Search that leaf
4. Optionally **backtrack** to neighboring regions (for better recall)

This is the foundation for KD-trees, ball trees, VP-trees, and others.

---

## 8. Two Fundamentally Different Ways to Cut Space

This distinction matters a lot for understanding the indexes that follow.

### A. Axis-Aligned Partitioning

Cut along **one dimension at a time**. Regions are rectangles/boxes.

```
Split on x=5:
  left: x < 5
  right: x >= 5

Then split left on y=3:
  left-bottom: x < 5, y < 3
  left-top: x < 5, y >= 3
```

**Pros:** Simple, fast boundary checks
**Cons:** Ignores correlations between dimensions

> This leads to **KD-trees**.

### B. Distance-Based Partitioning

Partition by **distance to reference points**. Regions are spheres or shells.

```
Pick pivot p, radius r:
  inside: dist(point, p) <= r
  outside: dist(point, p) > r
```

**Pros:** More geometry-aware, works when individual axes are meaningless
**Cons:** More expensive to evaluate (requires distance computation)

> This leads to **ball trees** and **VP-trees**.

---

## 9. The Hidden Assumption (Why This Will Break Later)

All spatial partitioning assumes:

> "Nearby points are meaningfully closer than far points."

In low dimensions this holds strongly — regions have clear separation, boundaries are meaningful.

**In high dimensions:**
- Distances concentrate (all pairwise distances become similar)
- Boundaries lose separation (regions overlap in practice)
- You end up searching most regions anyway

The partition exists, but it stops **pruning effectively**. This is the curse of dimensionality, and it's why pure tree-based methods struggle beyond ~20-30 dimensions.

This problem is what eventually motivates approaches like **IVF** — which still partitions space, but uses learned centroids instead of geometric cuts, and accepts approximation from the start.

---

## Key Takeaways

1. **Partitioning = divide space into regions** to avoid brute-force search
2. **Recursive partitioning = trees** — a natural and intuitive structure
3. **Axis-aligned vs distance-based** — two fundamentally different cut strategies
4. **Boundary problem is unavoidable** — nearest neighbors can always cross borders
5. **Works well in low dimensions**, degrades in high dimensions
6. **This is the setup** — specific indexes (KD-tree, ball tree) are implementations of these ideas