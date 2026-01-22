# Graph-Based ANN: Greedy Search Foundation — Notes

## 1. Problem Setup

**Given:**
- Huge set of vectors
- Query vector
- Want closest vector(s)

**Constraint:**
- Cannot check everything (too slow)
- Allowed to be approximately correct

**Solution:** Build a graph

---

## 2. Proximity Graph

**Structure:**
- Each vector = node
- Each node connects to small number of nearby nodes
- Each node has **few edges** (constant degree), not thousands

**Analogy:**
- Cities → nearby cities
- People → friends
- Wikipedia → related pages

---

## 3. Greedy Graph Search (Core Algorithm)
```text
1. Start from some node
2. Look at all its neighbors
3. Move to neighbor closer to query
4. Repeat
5. Stop when no neighbor is closer
```

**Characteristics:**
- No backtracking
- No global view
- Just "always step downhill"

---

## 4. Why It Works (Despite Looking Fragile)

**Concerns:**
- Start far away?
- Get stuck?
- Graph messy?

**Key fact:**
> In high-quality proximity graphs, local improvements usually lead globally closer

**Reason:** Semantic vector spaces are **not random**

---

## 5. Geometric Intuition

**Embedding spaces have:**
- Similar things cluster
- Clusters overlap smoothly
- Many short paths between related items

**Example path:**
```text
"dog" → "puppy" → "golden retriever"
```

**Key assumption:**
> **Nearest neighbors of neighbors are often even closer**

This is what graph ANN exploits.

---

## 6. Mental Picture: Hiking in Fog

- Can't see whole mountain
- Only see nearby terrain
- Always step downhill

**Works when landscape is:**
- Smooth (few sharp pits)
- Well-connected

**Assumption:**
> Embedding space = smooth landscape with shallow slopes, not random cliffs

---

## 7. Strengths

- Extremely fast
- Touches very few nodes
- No global computation
- Scales to millions/billions

**Replaces:** "Check everyone" → "Follow promising paths"

---

## 8. Failure Modes

### 1. Local Minima
Node's neighbors all worse, but global best elsewhere

### 2. Poorly Connected Graph
Too few edges → trapped

### 3. Bad Starting Point
Start far away → slow/miss regions

> **Graph design matters more than greedy rule itself**

---

## 9. Why Plain Greedy Isn't Enough

**Pure greedy:**
- Keeps only one active path
- Makes irreversible decisions
- Fragile if graph imperfect

**Modern algorithms add:**
- Multiple starting points
- Small candidate sets
- Backtracking within bounded window

**But heart is still greedy stepping**

---

## 10. Core Principles (Memorize)

1. Graph ANN does **navigation, not partitioning**
2. Greedy search follows **local improvements**
3. Works because embeddings have **local semantic continuity**
4. Failures from **poor graph structure**, not greediness
5. All modern graph ANN = **controlled greedy search**

---

## 11. Critical Understanding Check

**Q:** Why doesn't greedy search work on random graph?

**A:** 
- Neighbors give no directional signal
- No correlation: "closer now" ≠ "closer later"
- Wander aimlessly

> **Graph ANN works only because graph reflects geometry**

---

## Key Takeaway

Greedy graph search works because:
- Semantic spaces have local structure
- Good neighbors lead to better neighbors
- Graph encodes geometric proximity

**Failures come from graph quality, not the search strategy**
