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

What is the "_" for?
    It's a word-boundary marker. We add it during pre-tokenization
    so BPE knows where spaces were in the original text.

    Without it, BPE can't tell the difference between:
        "low" as a standalone word    → l o w _
        "low" inside "flower"         → ... l o w e r ...

    The "_" prevents the standalone "low" from merging with
    characters that belong to the next word.

    NOT every token in the final vocabulary ends with "_".
    The vocabulary contains all types of pieces:
        "low_"  → a complete word (has the end marker)
        "er_"   → a word ending (suffix + boundary)
        "low"   → a beginning or middle piece (no boundary)
        "er"    → a piece that could appear anywhere
        "e"     → a single character

    Tokens with "_" can only match at the END of a word.
    Tokens without "_" can match anywhere inside a word.
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

**"But with 32K merges, does it replay all 32K rules for every word?"**

```text
No. The merge rules are stored in a lookup table (hash map), not a list
you scan through one by one.

What gets saved after training:
    {
        ("l", "o")   → priority 1,      (learned first = highest priority)
        ("lo", "w")  → priority 2,
        ("e", "r")   → priority 3,
        ("er", "_")  → priority 4,
        ("low", "_") → priority 5,
        ...
        ("th", "e")  → priority 8,042,
        ("es", "t")  → priority 12,500,
        ...
    }
    A hash map with 32,000 entries. Looking up any pair is instant.

How tokenization actually works for "lowest":

    1. Split: [l, o, w, e, s, t]

    2. Check ALL adjacent pairs against the hash map:
         (l, o)  → found! priority 1
         (o, w)  → found! priority 47
         (w, e)  → found! priority 3,200
         (e, s)  → found! priority 12,500
         (s, t)  → found! priority 5,001
       That's 5 lookups, not 32,000.

    3. Apply the pair with the LOWEST priority number (= learned earliest):
         (l, o) → "lo"    →  [lo, w, e, s, t]

    4. Only re-check the pairs AFFECTED by this merge:
         (lo, w) → found! priority 2     ← new pair created by the merge
         (w, e)  → still priority 3,200  ← unchanged
         (no need to re-check (e,s) or (s,t) — they weren't affected)

    5. Apply lowest: (lo, w) → "low"  →  [low, e, s, t]

    6. Repeat until no adjacent pair exists in the hash map.

    For a 6-character word: ~5 lookups per round, ~4 rounds = ~20 lookups.
    NOT 32,000 sequential scans.

This is why tokenization is essentially instant — even with a 128K vocabulary.
```

---

## Byte-Level BPE (GPT-2 and onward)

Standard BPE operates on characters. But what counts as a "character"? Accents, Chinese characters, emoji — the Unicode space is huge. To understand how byte-level BPE solves this, you need to know what Unicode and UTF-8 are.

### Quick Refresher: Unicode and UTF-8

```text
The problem: computers store numbers, not letters.
We need a system that says "number 65 = the letter A."

PHASE 1 — ASCII (1960s):
    Assigns numbers 0-127 to English characters.
        65 = 'A'    97 = 'a'    48 = '0'    32 = ' '    10 = newline

    Each character fits in 1 byte (8 bits, values 0-255).
    Works great for English. Useless for everything else.

PHASE 2 — Unicode (1991):
    One giant table that assigns a number to EVERY character
    in EVERY language (plus symbols, emoji, ancient scripts, etc.)

    These numbers are called "code points."
    Unicode currently defines ~150,000 characters.

    But Unicode is just the TABLE — it says '猫' = number 29,483.
    It doesn't say how to store that number as bytes in memory.
    That's where encoding comes in.
```

### Hex Crash Course (needed for everything below)

```text
Why hex? Computers work in bytes. 1 byte = 8 bits = values 0-255.
Decimal is clunky for bytes: "195" doesn't tell you anything about bits.
Hex is compact: each hex digit = exactly 4 bits, so 1 byte = exactly 2 hex digits.

How hex works:
    Decimal counts:  0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16  17 ...
    Hex counts:      0  1  2  3  4  5  6  7  8  9   A   B   C   D   E   F  10  11 ...

    After 9, hex uses A-F for 10-15. Then 10 in hex = 16 in decimal.

Converting decimal → hex (just divide by 16 repeatedly):

    Example: 233 → hex
        233 ÷ 16 = 14 remainder 9
                    14 = E, 9 = 9
        → 0xE9

    Example: 29,483 → hex
        29,483 ÷ 16 = 1,842 remainder 11 (B)
         1,842 ÷ 16 =   115 remainder  2
           115 ÷ 16 =     7 remainder  3
             7 ÷ 16 =     0 remainder  7
        Read remainders bottom-up: 7, 3, 2, B → 0x732B

You don't need to memorise this. Just know:
    - "0x" or "U+" prefix means "the following digits are hex"
    - Each hex digit = 4 bits, two hex digits = 1 byte
    - U+0041 and 0x41 and 65 decimal are all the same number
```

### Unicode Code Points — The Lookup Table

```text
Unicode is just a giant agreed-upon table:

    Number (decimal)  │  Hex (U+ notation)  │  Character
    ──────────────────┼─────────────────────┼───────────
    65                │  U+0041             │  A
    97                │  U+0061             │  a
    233               │  U+00E9             │  é
    29,483            │  U+732B             │  猫
    128,049           │  U+1F431            │  🐱

That's it. Unicode says "character number 29,483 is 猫."
Written in hex: U+732B (because 29,483 in hex = 732B).

The U+ notation just means "Unicode code point, written in hex."
It's the same number, different notation.
The table goes up to U+10FFFF (= 1,114,111 in decimal).
```

### UTF-8 — How to Store Code Points as Bytes

```text
The problem:
    Unicode gives every character a number. But how do you store it?

    'A' = 65. Easy — fits in 1 byte.
    '猫' = 29,483. Doesn't fit in 1 byte (max 255). Needs multiple bytes.
    '🐱' = 128,049. Even bigger.

    Option A — Fixed width: always use 4 bytes per character.
        Simple, but "hello" = 20 bytes instead of 5. Wasteful.

    Option B — UTF-8: variable width.
        Small numbers → fewer bytes. Big numbers → more bytes.
        This is what the world actually uses (~98% of web pages).
```

```text
UTF-8 ENCODING RULES:

    UTF-8 uses templates. Based on how big the code point number is,
    pick a template, stuff the binary bits into the x slots:

    Code point range          │ Template (binary)              │ Bytes │ Data bits
    ──────────────────────────┼────────────────────────────────┼───────┼──────────
    0 – 127                   │ 0xxxxxxx                      │   1   │   7
    128 – 2,047               │ 110xxxxx  10xxxxxx            │   2   │  11
    2,048 – 65,535            │ 1110xxxx  10xxxxxx  10xxxxxx  │   3   │  16
    65,536 – 1,114,111        │ 11110xxx  10xxxxxx  10xxxxxx  10xxxxxx │ 4 │ 21

    The non-x bits are MARKERS:
        0_______ = "I'm a 1-byte character"
        110_____ = "I'm the START of a 2-byte character"
        1110____ = "I'm the START of a 3-byte character"
        11110___ = "I'm the START of a 4-byte character"
        10______ = "I'm a CONTINUATION byte (not a start)"

    These markers are how a computer reading bytes knows where
    one character ends and the next begins.
```

### Complete Example 1: Encoding 'é'

```text
Step 1: look up the code point
    'é' = U+00E9 = 233 in decimal

Step 2: which range?
    233 is in range 128–2,047 → use the 2-byte template
    Template: 110xxxxx  10xxxxxx   (11 data bit slots)

Step 3: convert 233 to binary
    233 = 128 + 64 + 32 + 8 + 1
        = 11101001 in binary (8 bits)
    Pad to 11 bits: 00011101001

Step 4: stuff the bits into the template
    Template:   110xxxxx   10xxxxxx
    Data bits:     00011     101001
    Result:     11000011   10101001

Step 5: convert each byte to decimal
    11000011 = 128+64+0+0+0+0+2+1 = 195
    10101001 = 128+0+32+0+8+0+0+1 = 169

RESULT: 'é' → UTF-8 bytes [195, 169]

    Verify: these are the bytes a computer stores when you type 'é'.
    Two bytes for one character.
```

### Complete Example 2: Encoding '猫'

```text
Step 1: look up the code point
    '猫' = U+732B = 29,483 in decimal

Step 2: which range?
    29,483 is in range 2,048–65,535 → use the 3-byte template
    Template: 1110xxxx  10xxxxxx  10xxxxxx   (16 data bit slots)

Step 3: convert 29,483 to binary
    29,483 = 0111 0011 0010 1011  (16 bits)

Step 4: stuff the bits into the template
    Template:   1110xxxx   10xxxxxx   10xxxxxx
    Data bits:      0111     001100     101011
    Result:     11100111   10001100   10101011

Step 5: convert each byte to decimal
    11100111 = 231
    10001100 = 140
    10101011 = 171

RESULT: '猫' → UTF-8 bytes [231, 140, 171]

    Three bytes for one character.
```

### Complete Example 3: Encoding '🐱'

```text
Step 1: look up the code point
    '🐱' = U+1F431 = 128,049 in decimal

Step 2: which range?
    128,049 is in range 65,536–1,114,111 → use the 4-byte template
    Template: 11110xxx  10xxxxxx  10xxxxxx  10xxxxxx   (21 data bit slots)

Step 3: convert 128,049 to binary
    128,049 = 0 00011 111010 000110 001  (padded to 21 bits)
    Let me regroup: 000 011111 010000 110001

Step 4: stuff the bits into the template
    Template:   11110xxx   10xxxxxx   10xxxxxx   10xxxxxx
    Data bits:       000     011111     010000     110001
    Result:     11110000   10011111   10010000   10110001

Step 5: convert each byte to decimal
    11110000 = 240
    10011111 = 159
    10010000 = 144
    10110001 = 177

RESULT: '🐱' → UTF-8 bytes [240, 159, 144, 177]

    Four bytes for one emoji.
```

### The Summary So Far

```text
Character → Unicode table → code point number → UTF-8 template → bytes

    'A'  → U+0041 →     65 → [65]                     (1 byte)
    'é'  → U+00E9 →    233 → [195, 169]               (2 bytes)
    '猫' → U+732B → 29,483 → [231, 140, 171]          (3 bytes)
    '🐱' → U+1F431→128,049 → [240, 159, 144, 177]     (4 bytes)

    English is cheap (1 byte/char). Emoji is expensive (4 bytes/char).
    But EVERYTHING is representable — no character is ever "unknown."

Why this matters for tokenization:
    These byte values (65, 195, 169, 231, ...) are what byte-level BPE
    uses as its starting vocabulary. Every possible text becomes a
    sequence of numbers 0-255, and BPE merges build up from there.
```

### Now Back to Byte-Level BPE

With that context, the jump to byte-level BPE is simple:

**Byte-level BPE** starts with **bytes** (0-255) instead of characters:

```text
Standard BPE base vocabulary: all Unicode characters (~150,000)
    Problem: huge initial vocabulary, many rare characters

Byte-level BPE base vocabulary: all single bytes (256)
    Every possible text can be represented as a sequence of bytes.
    No "unknown token" is ever needed.

Trace through the UTF-8 bytes from above:

"café" in bytes:  [99, 97, 102, 195, 169]     ← 'c','a','f' are 1 byte each
                                                   'é' is 2 bytes (195, 169)
"猫"   in bytes:  [231, 140, 171]              ← 3 bytes (CJK range)
"🐱"   in bytes:  [240, 159, 144, 177]         ← 4 bytes (emoji range)

Then BPE merges operate on these bytes → common byte sequences become tokens.
Frequent English words get merged into single tokens:
    "the" → bytes [116, 104, 101] → merged into one token early on
Rare scripts stay as byte sequences (more tokens, but never unknown):
    "猫" → bytes [231, 140, 171] → might stay as 3 separate byte tokens,
    or if Chinese is common in training data, the 3 bytes get merged into one token
```

This is why GPT can handle any language, emoji, or even binary data — at the byte level, everything is representable. The 256-byte base vocabulary is tiny and universal, and BPE merges build up from there based on what's frequent in the training data.

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
