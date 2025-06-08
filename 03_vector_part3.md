### Small code overview

Where the vectors come from today?

You can train an embedding model yourself, but most teams start with:
- API—OpenAI’s text-embedding-3-small,
- Cohere’s embed-v3,
- Google’s textembedding-gecko, or
- with open-source checkpoints like all-MiniLM-L6-v2.

Either way the recipe is the same: 
- pass your content to the model,
- receive the embedding vector,
- upsert it into the vector DB alongside a primary key and any metadata you need at retrieval time.
- Because the embedding function is deterministic, the same input will always yield the same vector, allowing you to rebuild or extend the index whenever model upgrades appear.

```python
# mini word2vec on a toy corpus ---------------------------------
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

corpus = [
    "the cat sat on the mat",
    "the dog lay on the rug",
    "cats and dogs are pets",
    "the quick brown fox jumps over the lazy dog",
]
sentences = [simple_preprocess(line) for line in corpus]

w2v = Word2Vec(
    sentences=sentences,
    vector_size=50,
    window=3,
    min_count=1,
    workers=2,
    sg=1,          # 1 = skip-gram, 0 = CBOW
    negative=5,
    epochs=100,
)

print("cosine(cat, dog) =", w2v.wv.similarity("cat", "dog"))
print("cosine(cat, fox) =", w2v.wv.similarity("cat", "fox"))

# use a public embedding model for arbitrary sentences ----------
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "The cat sat on the mat.",
    "A dog slept on the rug.",
    "Quantum entanglement defies classical intuition.",
    "Tacos taste best with fresh salsa.",
]

emb = model.encode(sentences, normalize_embeddings=True)
for i, s in enumerate(sentences):
    for j in range(i + 1, len(sentences)):
        sim = util.cos_sim(emb[i], emb[j]).item()
        print(f"{i}-{j}: cos={sim: .3f}")
```

```bash
cosine(cat, dog) = 0.16750851
cosine(cat, fox) = 0.1281476

0-1: cos= 0.468
0-2: cos= 0.031
0-3: cos= 0.053
1-2: cos= 0.049
1-3: cos=-0.119
2-3: cos=-0.014
```

Note:
the first block shows that the mini model already puts “cat” and “dog” closer than “cat” and “fox”. 
The second block produces four 384-D sentence vectors and prints pairwise cosine scores.
the two pet sentences cluster; the physics and taco sentences sit far apart, demonstrating how an off-the-shelf encoder generalises to anything you throw at it.
