## Tokenization

### Why Tokenization Matters

A transformer doesn't see text. It sees **numbers**. Tokenization is the step that converts text into numbers — and it's more important than it sounds.

```text
You might think: just split on spaces and assign IDs.

    "The cat sat" → ["The", "cat", "sat"] → [1, 2, 3]

But this breaks immediately:
    - "unhappiness" → one token? Or "un" + "happiness"? Or "un" + "happy" + "ness"?
    - "ChatGPT" → never seen this word before → ??? (unknown token)
    - "café" → what about accents?
    - "こんにちは" → Japanese has no spaces
    - "2024-03-19" → is this one token or five?

Tokenization determines:
    1. How the model sees your text (what units it thinks in)
    2. How long your input is (token count, not word count)
    3. What the model CAN'T understand (if a word gets split badly)
    4. How expensive your API call is (you pay per token)
```

---

## The Spectrum of Tokenization

```text
Character-level          Subword (BPE)              Word-level
─────────────────────────────────────────────────────────────
"cat" → [c, a, t]       "cat" → [cat]              "cat" → [cat]
3 tokens                 1 token                     1 token

"unhappiness"            "unhappiness"               "unhappiness"
→ [u,n,h,a,p,p,         → [un, happi, ness]         → [unhappiness]
   i,n,e,s,s]              or [un, happiness]           1 token
12 tokens                2-3 tokens

"xyzzy123"               "xyzzy123"                  "xyzzy123"
→ [x,y,z,z,y,1,2,3]     → [xy, zzy, 123]            → [UNK] ← UNKNOWN!
8 tokens                 3 tokens                     Can't handle it

Vocabulary size:          Vocabulary size:            Vocabulary size:
~256 (all bytes)          ~32K-100K                   ~100K-500K
Very small                Sweet spot                  Very large

Sequences very long       Reasonable length            Short
(slow attention)          (efficient)                  (but can't handle new words)
```

**Every modern LLM uses subword tokenization** — it's the sweet spot between character-level (too granular, sequences too long) and word-level (can't handle new/rare words).

---

## Byte Pair Encoding (BPE) — The Dominant Method

Used by: GPT-2, GPT-3, GPT-4, LLaMA, Mistral, Claude

### The Core Idea

Start with individual characters. Repeatedly merge the most frequent pair of adjacent tokens into a new token. Stop when you reach your desired vocabulary size.

### Building a BPE Vocabulary: Step-by-Step

**Training corpus** (the text we'll learn from):

```text
"low low low low low lowest lowest newer newer newer wider wider"
```

**Step 0: Start with characters**

Split every word into characters, add a special end-of-word marker `_`:

```text
l o w _       (appears 5 times — from "low" × 5)
l o w e s t _ (appears 2 times — from "lowest" × 2)
n e w e r _   (appears 3 times — from "newer" × 3)
w i d e r _   (appears 2 times — from "wider" × 2)

Initial vocabulary: {l, o, w, e, s, t, n, r, i, d, _}  (11 tokens)
```

**Step 1: Find the most frequent adjacent pair**

```text
Count all adjacent pairs across the corpus:
    (l, o) → 5 + 2 = 7 times
    (o, w) → 5 + 2 = 7 times
    (w, _) → 5 times
    (w, e) → 3 + 2 = 5 times
    (e, r) → 3 + 2 = 5 times
    (r, _) → 3 + 2 = 5 times
    (e, s) → 2 times
    (s, t) → 2 times
    ...

Most frequent: (l, o) with 7 occurrences.
Merge "l" + "o" → "lo"
```

**Step 2: Replace all occurrences and repeat**

```text
After merge 1 — "lo" added to vocabulary:
    lo w _       (5 times)
    lo w e s t _ (2 times)
    n e w e r _  (3 times)
    w i d e r _  (2 times)

Next most frequent: (lo, w) → 7 times
Merge "lo" + "w" → "low"

After merge 2 — "low" added:
    low _        (5 times)
    low e s t _  (2 times)
    n e w e r _  (3 times)
    w i d e r _  (2 times)

Next most frequent: (low, _) → 5 times, (e, r) → 5 times, (w, e) → 3 times...
Merge "e" + "r" → "er"

After merge 3 — "er" added:
    low _        (5 times)
    low e s t _  (2 times)
    n e w er _   (3 times)
    w i d er _   (2 times)

Next: (er, _) → 5 times
Merge "er" + "_" → "er_"

After merge 4 — "er_" added:
    low _        (5 times)
    low e s t _  (2 times)
    n e w er_    (3 times)
    w i d er_    (2 times)

Next: (low, _) → 5 times
Merge "low" + "_" → "low_"

After merge 5 — "low_" added:
    low_         (5 times)
    low e s t _  (2 times)
    n e w er_    (3 times)
    w i d er_    (2 times)
```

And so on. Each merge adds one new token to the vocabulary.

**After ~30,000 merges:**

```text
Vocabulary: {l, o, w, e, s, t, n, r, i, d, _, lo, low, er, er_,
             low_, est, new, newer_, ..., the, ing, tion, ...}

Common words → single tokens:     "the" → [the]
Common subwords → merged:          "running" → [runn, ing]
Rare words → character fallback:   "xyzzy" → [x, y, z, z, y]
```

### Using BPE to Tokenize New Text

At inference time, you don't retrain. You apply the learned merges in order:

```text
Input: "lowest"

1. Split into characters:  [l, o, w, e, s, t]
2. Apply merge rules in the order they were learned:
    Merge 1: (l, o) → lo    →  [lo, w, e, s, t]
    Merge 2: (lo, w) → low  →  [low, e, s, t]
    Merge 3: (e, s) → es    →  [low, es, t]
    Merge 4: (es, t) → est  →  [low, est]
    No more applicable merges.

Result: "lowest" → [low, est]    (2 tokens)

This is why "lowest" gets tokenized as [low, est] and not [lowe, st]:
the merges were learned in frequency order, and "low" was merged before "est".
```

---

## Byte-Level BPE (GPT-2 and onward)

Standard BPE operates on characters. But what counts as a "character"? Accents, Chinese characters, emoji — the Unicode space is huge.

**Byte-level BPE** starts with **bytes** (0-255) instead of characters:

```text
Standard BPE base vocabulary: all Unicode characters (~150,000)
    Problem: huge initial vocabulary, many rare characters

Byte-level BPE base vocabulary: all single bytes (256)
    Every possible text can be represented as a sequence of bytes.
    No "unknown token" is ever needed.

"café" in bytes:  [99, 97, 102, 195, 169]     (UTF-8 encoding)
"猫"   in bytes:  [231, 140, 171]              (UTF-8 encoding)
"🐱"   in bytes:  [240, 159, 144, 177]         (UTF-8 encoding)

Then BPE merges operate on these bytes → common byte sequences become tokens.
Frequent English words get merged into single tokens.
Rare scripts stay as byte sequences (more tokens, but never unknown).
```

This is why GPT can handle any language, emoji, or even binary data — at the byte level, everything is representable.

---

## WordPiece (BERT)

Very similar to BPE, with one key difference in how merges are chosen.

```text
BPE:       merge the most FREQUENT pair
WordPiece: merge the pair that maximizes LIKELIHOOD of the training data

In practice, the difference is subtle:
    BPE picks: "the pair I see most often"
    WordPiece picks: "the pair that makes my language model most accurate"

WordPiece also uses a "##" prefix for non-initial subwords:
    "unhappiness" → ["un", "##happi", "##ness"]

    The "##" tells you this piece is a continuation, not a word start.
    BPE doesn't use this — it relies on whitespace handling instead.
```

WordPiece is mainly a BERT thing. Most modern LLMs (GPT, LLaMA, Claude) use BPE.

---

## SentencePiece (LLaMA, Mistral, Gemini)

The problem with standard BPE: it assumes whitespace-separated words (split on spaces first, then BPE within words). This breaks for languages without spaces (Chinese, Japanese, Thai).

```text
Standard BPE:
    1. Pre-tokenize: split on spaces → ["The", "cat", "sat"]
    2. Apply BPE within each word

    Problem: "我喜欢猫" has no spaces. Step 1 produces ["我喜欢猫"].
    BPE can only merge within this one giant string.

SentencePiece:
    Treats the raw text as a sequence of characters (or bytes).
    No pre-tokenization step. No assumption about spaces.
    Spaces are treated as a regular character: "▁" (U+2581)

    "The cat sat" → "▁The▁cat▁sat"
    "我喜欢猫"    → "▁我▁喜欢▁猫"   (or "▁我喜欢猫" depending on training)

    Then applies BPE (or Unigram) on this unified stream.
```

**SentencePiece also supports Unigram tokenization** — an alternative to BPE:

```text
BPE (bottom-up):
    Start with characters → merge up to vocabulary size
    Greedy: once a merge is made, it's permanent

Unigram (top-down):
    Start with a LARGE vocabulary of all possible subwords
    Iteratively REMOVE the least useful tokens
    Keep the set that maximizes the likelihood of the training data

    More principled (information-theoretic) but results are similar to BPE.
    LLaMA uses SentencePiece with BPE. Some models use SentencePiece with Unigram.
```

---

## Vocabulary Size: The Trade-Off

```text
Small vocabulary (e.g., 256 — just bytes):
    ✓ Every possible input can be tokenized
    ✗ Sequences are very long ("hello" = 5 tokens)
    ✗ Attention is O(n²) — long sequences are expensive
    ✗ Model must learn to assemble meaning from tiny pieces

Large vocabulary (e.g., 500K — many whole words):
    ✓ Short sequences (each word = 1 token)
    ✓ Model sees meaningful units directly
    ✗ Embedding table is huge (500K × 768 = 384M params just for embeddings)
    ✗ Rare words still get [UNK]
    ✗ Wastes capacity on very rare tokens

Sweet spot (~32K - 100K):
    Common words → 1 token:     "the" → [the]
    Common subwords → merged:   "running" → [runn, ing]
    Rare words → subword pieces: "defenestration" → [def, en, est, ration]
    Never fails on any input    (byte-level fallback)
```

Typical vocabulary sizes:

```text
| Model        | Vocab Size | Method                    |
| ------------ | ---------- | ------------------------- |
| GPT-2        | 50,257     | Byte-level BPE            |
| GPT-4        | ~100,000   | Byte-level BPE (tiktoken) |
| BERT         | 30,522     | WordPiece                 |
| LLaMA        | 32,000     | SentencePiece BPE         |
| LLaMA 3      | 128,256    | Byte-level BPE (tiktoken) |
| Gemini       | 256,000    | SentencePiece             |
| Mistral      | 32,000     | SentencePiece BPE         |
```

The trend is growing vocabularies — LLaMA 3 jumped from 32K to 128K. Larger vocabularies mean shorter sequences (fewer tokens per text), which means faster inference (less attention computation per generation step). The cost is a larger embedding table.

---

## Special Tokens

Every tokenizer includes tokens that aren't real words:

```text
| Token    | Meaning                | Used by           |
| -------- | ---------------------- | ----------------- |
| <BOS>    | Beginning of sequence  | Decoders (GPT)    |
| <EOS>    | End of sequence        | All models        |
| <PAD>    | Padding (fill batches) | All models        |
| <UNK>    | Unknown token          | WordPiece (BERT)  |
|          |                        | Not needed in BPE |
| [CLS]    | Classification token   | BERT              |
| [SEP]    | Separator              | BERT              |
| [MASK]   | Masked token           | BERT (for MLM)    |

Chat models add more:
| <|system|>     | Start of system prompt    |
| <|user|>       | Start of user message     |
| <|assistant|>  | Start of assistant reply  |
| <|end_turn|>   | End of a turn             |

These structure the conversation:
    <|system|>You are a helpful assistant.<|end_turn|>
    <|user|>What is 2+2?<|end_turn|>
    <|assistant|>4.<|end_turn|>
```

The model learns what these tokens mean during training — `<EOS>` means "stop generating," `<|user|>` means "the human is speaking now."

---

## The Tokenization Pipeline (Full Example)

Let's trace "I don't like ChatGPT's tokenizer!" through GPT-2's tokenizer:

```text
Step 1: Pre-tokenization (split on whitespace and punctuation boundaries)
    → ["I", " don", "'t", " like", " Chat", "G", "PT", "'s", " token", "izer", "!"]

    Note: spaces are attached to the NEXT word (" don", " like", " Chat")
    This is GPT-2's convention — spaces are part of the token.

Step 2: Apply BPE merges to each piece
    "I"       → [I]              (common word, single token)
    " don"    → [Ġdon]           (Ġ represents a leading space)
    "'t"      → ['t]             (common contraction, single token)
    " like"   → [Ġlike]          (single token)
    " Chat"   → [ĠChat]         (single token)
    "G"       → [G]              (single character)
    "PT"      → [PT]             (merged — common pair)
    "'s"      → ['s]             (single token)
    " token"  → [Ġtoken]        (single token)
    "izer"    → [izer]           (common suffix, single token)
    "!"       → [!]              (single character)

Step 3: Map to token IDs
    [I, Ġdon, 't, Ġlike, ĠChat, G, PT, 's, Ġtoken, izer, !]
    → [40, 836, 470, 588, 8537, 38, 11571, 338, 11241, 7509, 0]

    11 tokens for 6 words.
```

### What the Model Actually Sees

```text
The model never sees the text "I don't like ChatGPT's tokenizer!"
It sees: [40, 836, 470, 588, 8537, 38, 11571, 338, 11241, 7509, 0]

Each ID looks up a 768-d vector in the embedding table.
The transformer processes these 11 vectors.
It has no concept of "words" — only tokens.
```

---

## Why Tokenization Creates Weird Behaviour

### The "How many r's in strawberry?" Problem

```text
"strawberry" gets tokenized as: [str, aw, berry]

The model sees 3 tokens: "str", "aw", "berry"
It never sees the individual letters!

To count r's, the model would need to:
    1. Know that "str" contains an "r"
    2. Know that "aw" contains no "r"
    3. Know that "berry" contains two "r"s
    4. Add them up: 1 + 0 + 2 = 3

This is hard because the model learned "str" as an atomic unit.
It doesn't naturally decompose tokens into characters.
It's like asking you: "How many strokes are in this Chinese character: 龍?"
You see one symbol, not the individual strokes.
```

### Arithmetic Difficulty

```text
"What is 1234 + 5678?"

Tokenization: [12, 34, +, 56, 78]

The model sees "12" and "34" as separate tokens.
It must figure out they form "1234" and align digits for addition.
This is why LLMs struggle with arithmetic — the tokenizer
splits numbers in ways that don't respect digit positions.

Some models (LLaMA 3) tokenize each digit separately:
    [1, 2, 3, 4, +, 5, 6, 7, 8]
    This makes arithmetic easier but creates longer sequences.
```

### Language Inequality

```text
English text:
    "Hello, how are you?" → 6 tokens

Korean text (same meaning):
    "안녕하세요, 어떻게 지내세요?" → 15 tokens

Japanese text (same meaning):
    "こんにちは、お元気ですか？" → 12 tokens

Why? The BPE vocabulary was trained on mostly English text.
English words got merged into single tokens.
Korean/Japanese characters stayed as smaller pieces.

Consequences:
    - Non-English users hit context limits faster (more tokens per message)
    - Non-English users pay more per API call (billed per token)
    - The model has less "thinking capacity" per concept in non-English
      (each concept takes more token positions)

This is improving — multilingual models (LLaMA 3, Gemini) train tokenizers
on diverse multilingual data, giving better coverage to non-English scripts.
```

---

## Tokenizer ≠ Model

The tokenizer is trained **separately** from the model and **before** the model:

```text
Step 1: Train the tokenizer
    Input: large text corpus
    Output: vocabulary + merge rules
    Algorithm: BPE / WordPiece / Unigram
    This is a simple statistical algorithm, not a neural network.
    Takes minutes to hours.

Step 2: Train the model
    Input: text tokenized using Step 1's tokenizer
    Output: a trained transformer
    The model learns to predict tokens, not characters or words.
    Takes weeks to months on thousands of GPUs.

Once the tokenizer is trained, it's FROZEN.
The model is designed around that specific tokenizer.
You can't swap tokenizers after training — the embedding table,
vocabulary size, and all learned representations depend on it.
```

This is why you sometimes see "tokenizer" as a separate download alongside a model — they're paired but distinct artifacts.

---

## How to Choose / Compare Tokenizers

```text
| Feature                 | BPE (GPT)          | WordPiece (BERT)  | SentencePiece        |
| ----------------------- | ------------------ | ----------------- | -------------------- |
| Merge criterion         | Most frequent pair | Max likelihood    | BPE or Unigram       |
| Handles all inputs      | Yes (byte-level)   | No ([UNK] token)  | Yes (byte fallback)  |
| Space handling          | Space = part of token | Pre-split       | Space = "▁" character |
| No-space languages      | Via byte fallback  | Poor              | Native support        |
| Used by                 | GPT, Claude        | BERT, RoBERTa     | LLaMA, Gemini, T5    |
```

**In practice, the differences between BPE and SentencePiece-BPE are small.** The bigger factor is vocabulary size and what training data the tokenizer saw. A tokenizer trained on diverse multilingual data will handle non-English better regardless of the algorithm.

---

## Summary

```text
1. Tokenization converts text → token IDs → embedding vectors
2. Modern LLMs use subword tokenization (BPE or SentencePiece)
3. BPE: start with characters/bytes, iteratively merge frequent pairs
4. Vocabulary size (~32K-128K) balances sequence length vs table size
5. The tokenizer is trained BEFORE the model and frozen
6. Tokenization explains many LLM quirks:
    - Can't count letters in words (tokens ≠ characters)
    - Struggles with arithmetic (numbers split weirdly)
    - Non-English is more expensive (fewer merges → more tokens)
7. Special tokens (<BOS>, <EOS>, <|user|>) structure the input
```
