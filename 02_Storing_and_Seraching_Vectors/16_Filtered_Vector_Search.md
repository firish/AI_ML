# Filtered Vector Search (Metadata Filtering) — Notes

## 0. The Problem

Real queries are almost never "find the most similar vector." They're:

> "Find the most similar **shoes** under **$50** in **size 10**"

That's a vector similarity search **plus** metadata filters. Every production vector DB needs this.

```
Query = semantic similarity (vector) + hard constraints (metadata)
```

The challenge: ANN indexes (HNSW, IVF) are built for pure vector search. Filters break their assumptions.

---

## 1. What Metadata Looks Like

Each vector has associated key-value metadata:

```
vector_42:
  embedding: [0.2, 0.5, ..., 0.1]    (768-dim)
  metadata:
    category: "shoes"
    price: 49.99
    size: 10
    brand: "Nike"
    in_stock: true
```

**Filter types:**
- Equality: `category = "shoes"`
- Range: `price < 50`
- Set membership: `brand IN ["Nike", "Adidas"]`
- Boolean: `in_stock = true`
- Compound: `category = "shoes" AND price < 50 AND in_stock = true`

---

## 2. The Three Strategies

There are only three ways to combine vector search with filtering:

### Strategy A: Post-filtering

1. Run vector search normally (ignore filters)
2. Get top-N results
3. Filter out results that don't match metadata
4. Return what's left

### Strategy B: Pre-filtering

1. Apply metadata filter first → get set of valid vector IDs
2. Run vector search only on those IDs

### Strategy C: In-filter (integrated)

1. During vector search, check metadata at each step
2. Skip vectors that don't match filters
3. Continue searching until you have k valid results

Each has sharp tradeoffs. Let's go through them.

---

## 3. Post-Filtering

```python
function POST_FILTER_SEARCH(q, k, filters, index):
    """
    q       : query vector
    k       : desired number of results
    filters : metadata conditions
    index   : vector index (HNSW, IVF, etc.)
    N       : number of extra results to fetch (overquery)
    """
    # Step 1: fetch more than k results (to survive filtering)
    N = k * overquery_factor    # e.g., 10x
    raw_results = index.search(q, N)

    # Step 2: filter
    filtered = [r for r in raw_results if matches(r, filters)]

    # Step 3: return top-k from filtered
    return filtered[:k]
```

**Pros:**
- Simplest to implement
- Vector index doesn't need modification
- Works with any index type

**Cons:**
- If filter is selective (few vectors match), most results get thrown away
- You might not get k results at all
- Overquerying wastes compute

**When it breaks:**

```
Filter selectivity = 1%  (only 1% of vectors match)
Need k = 10 results
Must fetch ~1000 raw results to expect 10 survivors
Even that might not be enough (depends on distribution)
```

**Rule of thumb:** Post-filtering works when filter matches > 10-20% of data. Below that, it degrades fast.

---

## 4. Pre-Filtering

```python
function PRE_FILTER_SEARCH(q, k, filters, metadata_index, vectors):
    """
    q              : query vector
    k              : desired number of results
    filters        : metadata conditions
    metadata_index : traditional index on metadata (B-tree, bitmap, etc.)
    vectors        : all stored vectors
    """
    # Step 1: get matching IDs from metadata index
    valid_ids = metadata_index.query(filters)

    # Step 2: brute-force search only valid vectors
    results = []
    for id in valid_ids:
        d = dist(q, vectors[id])
        results.append((d, id))

    return top_k_closest(results, k)
```

**Pros:**
- Always returns exactly k results (if k valid vectors exist)
- Correct — doesn't miss valid results
- Works well with highly selective filters

**Cons:**
- Can't use the vector index — you're brute-forcing within the valid set
- If filter matches many vectors (low selectivity), brute force is slow
- Requires a separate metadata index

**When it breaks:**

```
Filter selectivity = 80% (most vectors match)
Valid set = 800K out of 1M
Brute-forcing 800K vectors ≈ just do brute force on everything
```

**Rule of thumb:** Pre-filtering works when filter matches < 5-10% of data. Above that, the valid set is too large for brute force.

---

## 5. The Selectivity Gap

Notice the problem:

```
Post-filtering:  works when filter matches MOST vectors (>10-20%)
Pre-filtering:   works when filter matches FEW vectors (<5-10%)
```

There's a gap in the middle where neither works great. This is why integrated approaches exist.

---

## 6. In-Filter (Integrated Filtering)

The idea: modify the vector search algorithm itself to skip non-matching vectors during traversal.

### With HNSW

```python
function FILTERED_HNSW_SEARCH(q, k, efSearch, filters):
    """
    Same as HNSW search, but skip nodes that fail the filter.
    Only count filter-passing nodes toward the k results.
    """
    # ... standard HNSW greedy descent on upper layers ...

    # At layer 0: modified bounded search
    while candidates not empty:
        (dc, c) = candidates.pop_min()

        # STILL TRAVERSE non-matching nodes (for navigation)
        for n in neighbors(c, layer=0):
            if n in visited:
                continue
            visited.add(n)
            dn = dist(q, n)

            if passes_filter(n, filters):
                # only ADD TO RESULTS if filter passes
                best.push((dn, n))

            # always add to candidates (for navigation)
            candidates.push((dn, n))

    return best.top_k(k)
```

**Critical detail:** Non-matching nodes are still **traversed** (used for navigation) but not **returned** (excluded from results). If you skip them entirely, you break the graph's connectivity and search quality collapses.

### With IVF

```python
function FILTERED_IVF_SEARCH(q, k, nprobe, filters):
    # Find closest clusters (unchanged)
    clusters = top_nprobe_clusters(q, nprobe)

    results = []
    for cluster in clusters:
        for v in cluster.vectors:
            if not passes_filter(v, filters):
                continue    # skip non-matching
            d = dist(q, v)
            results.append((d, v))

    return top_k_closest(results, k)
```

Simpler for IVF — just skip non-matching vectors during the list scan.

**Pros:**
- Works across all selectivity ranges
- Uses the vector index structure
- Most production systems use this approach

**Cons:**
- More complex implementation
- Filter checks at every node add overhead
- Very selective filters can cause the search to explore much more of the graph (in HNSW) before finding k valid results

---

## 7. How Production Systems Handle This

### Adaptive strategy selection

Most vector DBs don't use a single strategy. They estimate filter selectivity first, then choose:

```
Estimate: what fraction of vectors pass the filter?

if selectivity > 20%:
    use in-filter or post-filter
elif selectivity < 1%:
    use pre-filter (brute force on small valid set)
else:
    use in-filter with expanded search
```

### Bitmap indexes for metadata

Metadata filters are evaluated using **bitmap indexes** — one bit per vector per filter value:

```
category="shoes":   [1,0,0,1,1,0,0,1,...]    (1 = matches)
in_stock=true:      [1,1,0,1,0,1,1,1,...]

AND them together:  [1,0,0,1,0,0,0,1,...]    (both conditions)
```

Bitmap AND/OR operations are extremely fast (CPU word-level bitwise ops). This makes filter evaluation cheap even for complex compound filters.

### Partitioned indexes

Some systems build **separate indexes per filter value:**

```
index_shoes = HNSW(all shoe vectors)
index_shirts = HNSW(all shirt vectors)
```

Query "similar shoes" → search only `index_shoes`.

**Pros:** No filter overhead during search.
**Cons:** Index explosion if many filter values. Updates affect multiple indexes.

Works well when filter cardinality is low (few categories) and queries almost always filter on that field.

---

## 8. The Selective Filter Problem in HNSW

HNSW has a specific issue with very selective filters that's worth understanding.

**Normal HNSW:** Greedy search converges in O(log N) hops because each hop gets closer.

**With 1% selectivity filter:** You need to find k vectors that both (a) are near the query AND (b) pass the filter. Only 1 in 100 nodes qualifies.

**What happens:**
- Search reaches the right region quickly (HNSW is good at this)
- But most nodes in that region fail the filter
- Must explore many more nodes to find k valid ones
- Search expands outward, visiting nodes farther from the query
- Latency increases significantly

**Mitigations:**
- Increase `efSearch` for filtered queries
- Use pre-filtering if selectivity is known to be very low
- Build partitioned indexes for common filter fields
- Some systems (Qdrant, Weaviate) dynamically switch strategies based on estimated selectivity

---

## 9. Comparison Table

| Strategy | Best when | Worst when | Complexity |
|---|---|---|---|
| Post-filter | Broad filters (>20% match) | Selective filters | Simple |
| Pre-filter | Very selective (<5% match) | Broad filters | Need metadata index |
| In-filter | Middle ground, general purpose | Very selective on HNSW | Modify search algo |
| Partitioned index | Low-cardinality filter field | Many filter values | Multiple indexes |
| Adaptive | All cases | — | Most complex |

---

## Key Takeaways

1. **Real queries = vector similarity + metadata filters** — pure ANN is not enough
2. **Three strategies:** post-filter (simple, wastes compute), pre-filter (brute force valid set), in-filter (modify search)
3. **Selectivity determines the right strategy** — no single approach works for all filter rates
4. **In HNSW: still traverse non-matching nodes** — they're needed for navigation even if excluded from results
5. **Bitmap indexes** make filter evaluation cheap (bitwise AND/OR)
6. **Production systems adapt** — estimate selectivity, pick strategy per query
7. **Very selective filters are the hard case** — HNSW explores much more of the graph, latency spikes

---

**Next:** Hybrid search — combining dense vectors (semantic similarity) with sparse vectors (keyword matching / BM25).