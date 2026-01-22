## Part 1: k-NN Graph (Ideal World)

### Definition
- Every vector = node
- Each node connects to **k true closest vectors**
- "Closest" = exact nearest neighbors

**If infinite compute:** this is the perfect graph

---

### Why k-NN Graph is Perfect

**Property:**
> If node A is not closest to query, at least one neighbor is closer

**Result:**
- No bad local minima
- Many short paths
- Strong geometric signal
- Greedy search **must** reach true nearest neighbor

**k-NN graph = gold standard reference**

---

### Why We Don't Use It

**Build cost:** O(N² × d)
- For each of N nodes
- Compare to N-1 other nodes

**Example:** 10M vectors → impossible

**Status:**
- Conceptually perfect
- Computationally infeasible

**Key:** k-NN graphs explain **why** graph search works, not **how** to build efficiently

---

## Part 2: NSW (Navigable Small World)

### Problem NSW Solves

**Want:**
- Graph like k-NN
- Without O(N²) cost
- Built incrementally
- Good enough for greedy search

---

### Core Idea

**Instead of:**
> "Find my true k nearest neighbors"

**NSW says:**
> "Find some nearby neighbors and connect to them"

**Magic insight:** Don't need perfect neighbors — just reasonably close ones

---

### How NSW Builds the Graph

**Parameters (fixed):**
- `M`: target neighbors per node (10-40)
- `efConstruction`: search effort during insert (100-200)

**For each new vector X:**

1. **Pick ONE random entry point** (not k)
2. **Run bounded best-first search** toward X
   - Maintain candidate list (~efConstruction size)
   - Explore promising neighbors
   - Get ~100 decent candidates near X
3. **Select best M neighbors** from candidates
   - Sort by distance to X
   - Pick top M closest
4. **Connect X → neighbors**
5. **Bidirectional linking** (common)
   - Add neighbor → X
   - With degree limits + pruning

---

### Key Details

**NOT pure greedy:**
- Bounded best-first search
- Maintains candidate pool
- More robust than single-path

**Node degree:**
- **Not fixed** — bounded, not exact
- Nodes have **up to M** neighbors
- Some fewer, hubs may have more
- Soft limits for better connectivity

**Edge direction:**
- **Bidirectional = default** in practice
- Much better navigability
- Requires pruning logic

---

### Pruning (When Node Exceeds Degree)

Node Y has M neighbors, X wants to connect:
1. Consider Y's neighbors ∪ {X}
2. Keep M closest
3. Drop rest

**Crucial for quality**

---

### Why NSW Works

**Self-healing property:**
- Every insertion improves local geometry
- Later nodes benefit from earlier structure
- Approximate neighbors good enough
- Errors don't compound

**Creates small-world graph:**
- Any node reachable in few hops
- Local neighborhoods dense
- Global paths short

---

### NSW vs k-NN Graph

| Property | k-NN Graph | NSW |
|----------|------------|-----|
| Neighbors | Perfect | Approximate |
| Build cost | Too expensive | Incremental, cheap |
| Greedy success | Almost guaranteed | Very high, not perfect |
| Local minima | Rare | Possible |

---

### NSW Limitations (Why HNSW Needed)

1. **Single layer** — no "long jumps"
2. **Hard to escape local regions**
3. **Entry point dependent**
   - Bad start → worse results
4. **Limited tuning flexibility**
   - Recall-latency tradeoff constrained

---

### Parameter Effects

**Double M:**
- ✓ More memory
- ✓ Better connectivity
- ✗ Slower build
- ✓ Better recall

**Double efConstruction:**
- ✗ Slower insert
- ✓ Better neighbor choice
- ✓ Better final graph

---

## Mental Bridge to HNSW

1. **k-NN graph:** perfect structure, impossible at scale
2. **NSW:** approximate k-NN built incrementally via greedy search
3. **NSW proves:** greedy navigation works, but sometimes gets stuck
4. **HNSW:** adds hierarchy to fix NSW's weaknesses
