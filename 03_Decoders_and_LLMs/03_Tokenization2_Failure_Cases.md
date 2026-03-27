## Tokenization Failure Cases — When LLMs Get Simple Things Wrong

Tokenization (file 03) determines what the model "sees." When the task requires reasoning at a level finer than tokens — individual characters, digit positions, phonetics — models break in predictable ways. This file catalogues the most well-known failures, why they happen, and how models try to work around them.

---

## 1. Counting Characters in Words

### The Classic: "How many r's in strawberry?"

```text
"strawberry" gets tokenized as: [str, aw, berry]

The model sees 3 tokens. It never sees individual letters.
To count r's, it would need to:
    1. Know "str" contains one 'r'
    2. Know "aw" contains zero 'r's
    3. Know "berry" contains two 'r's
    4. Add: 1 + 0 + 2 = 3

This is hard because the model learned "str" as an atomic symbol.
It's like asking: "how many straight lines are in the letter 'A'?"
You see one symbol, not its internal geometry.

Common wrong answer: "2" (models often miss the 'r' hiding in "str")
Correct answer: 3
```

### The Harder Case: Random Strings

```text
"Count the u's in euf2heiuhcieucxe23uicbh3uye2rcgvoeevvrvrguutuuyy
tufblyeruhcveyu3h2hbvoyrtbcvuyr2ecxbcv"

This is MUCH harder than "strawberry" because:
    1. The string never appeared in training data
       (strawberry at least was seen spelled out millions of times)
    2. The tokenizer splits it into arbitrary chunks:
       [euf, 2he, iu, hc, ieu, cxe, 23, uic, bh, 3, uy, ...]
    3. The chunk boundaries have no relationship to where 'u' appears
    4. The model must mentally decompose each meaningless token into
       characters — for a long sequence with no familiar patterns

Even models that get "strawberry" right regularly fail here.
Different models give different wrong answers (13, 15, 16...)
because each one drops or double-counts at different token boundaries.
```

### How Models Try to Solve Character Counting

```text
Approach 1: Chain-of-thought (spell it out)
    The model generates the word character by character:
    "s-t-r-a-w-b-e-r-r-y. Let me count: r at position 3, r at 8, r at 9. That's 3."

    Why this works:
        Each letter the model GENERATES becomes its own token.
        "s", "t", "r", "a", ... → now the model can "see" individual letters
        because they ARE individual tokens in its output.

        The model re-encodes the word at a finer granularity than it received.
        It's not looking inside [str, aw, berry].
        It's reproducing a memorized spelling: [str,aw,berry] → s,t,r,a,w,b,e,r,r,y.

    Why it still fails on random strings:
        "strawberry" was seen spelled out millions of times in training data.
        "euf2heiuhcieucxe23..." was never seen anywhere.
        The model must decompose unfamiliar token chunks into characters
        ON THE FLY — and one skipped character in the middle means a
        confidently wrong answer with no way to catch the mistake.

Approach 2: Tool use (call Python)
    Model generates: "strawberry".count('r')
    Python returns: 3

    Python operates on actual characters, not tokens.
    It gets this right 100% of the time.
    This is the more reliable approach for any character-level task.

The honest truth:
    Models don't "see inside" tokens. They either:
    (a) Reproduce memorized spellings from training data (fragile), or
    (b) Offload to a tool that operates on characters (reliable)
```

---

## 2. Arithmetic

### Why 1234 + 5678 Is Hard

```text
"What is 1234 + 5678?"

Tokenization: [12, 34, Ġ+, Ġ56, 78]

The model sees "12" and "34" as SEPARATE tokens.
It doesn't naturally see "1234" as a four-digit number.
To add correctly, it needs to:
    1. Reconstruct that "12" + "34" = "1234" (not 12 and 34)
    2. Reconstruct that "56" + "78" = "5678"
    3. Align digits: thousands, hundreds, tens, ones
    4. Handle carries across token boundaries

    The carry from 4+8=12 happens at the boundary between
    tokens "34" and "12" — the model must carry across tokens.

This is why models get simple arithmetic wrong:
    7 + 8 = ?     → Usually correct (single-token numbers)
    234 + 567 = ? → Often correct (small enough to pattern-match)
    12847 + 9283 = ? → Frequently wrong (multi-token, multiple carries)
```

### Multi-Step Arithmetic

```text
"What is 15% of 847?"

The model needs to:
    1. Convert 15% to 0.15
    2. Multiply: 847 × 0.15
    3. Handle the decimal arithmetic correctly

Each step can introduce errors, and they compound.
The model isn't running a calculator — it's predicting the most
likely next token, which happens to sometimes look like arithmetic.

Common pattern:
    Correct reasoning, wrong final number.
    "15% of 847 = 847 × 0.15 = 127.05" ← correct
    "15% of 847 = 847 × 0.15 = 126.05" ← off by one, very common
```

### How Models Handle Arithmetic

```text
Chain-of-thought: break into steps, works for simple problems
    "1234 + 5678: ones: 4+8=12, carry 1. Tens: 3+7+1=11, carry 1..."
    But this is slow and still error-prone for large numbers.

Tool use: call a calculator or Python
    The reliable solution. 1234 + 5678 → Python returns 6912.

Training tricks:
    Some models (LLaMA 3) tokenize each digit separately:
    "1234" → [1, 2, 3, 4] instead of [12, 34]
    This makes arithmetic easier (each digit is its own token)
    but creates longer sequences (more tokens per number).
```

---

## 3. Reversing Words and Strings

```text
"Reverse the word 'hello'"

Tokenization: [hel, lo]  (NOT [h, e, l, l, o])

The model sees 2 chunks, not 5 characters.
To reverse, it needs to mentally decompose into characters first.

Common wrong answers:
    "olleh" ← correct (model memorized this common example)
    But try: "reverse 'algorithm'"
    Tokenized as: [alg, orith, m]
    Correct: "mhtirogla"
    Model might say: "mhtirogla" or "mhtirogla" (subtle transposition)

The failure pattern:
    - Common words (hello, world, python): usually correct (memorized)
    - Uncommon words: error rate increases
    - Random strings: very unreliable
    - Long strings: almost always wrong somewhere in the middle
```

---

## 4. Rhyming and Phonetics

```text
"What rhymes with 'thought'?"

The model sees tokens, not sounds.
"thought" → [th, ought] — no phonetic information in these tokens.

The model must have learned during training that:
    "ought" sounds like "ot" (not "ow" or "uff")
    "bought", "caught", "fought" all end in the same sound

This usually works for common words (learned from training data).
It fails for:
    - Uncommon words: "What rhymes with 'segue'?" (pronounced "seg-way")
    - Cross-language: English spelling ≠ pronunciation
    - Made-up words: "What rhymes with 'blought'?"
      Model can't sound it out — it must guess from token patterns.

The root cause:
    Text has no phonetic encoding. "thought" and "taut" rhyme,
    but their tokens share no characters. "thought" and "though"
    share most characters but DON'T rhyme.
    The model must learn pronunciation as implicit knowledge,
    not from the token structure itself.
```

---

## 5. Comparing String Lengths

```text
"Which is longer: 'extraordinarily' or 'communication'?"

Tokenized:
    "extraordinarily" → [extra, ordin, arily]      3 tokens
    "communication"   → [commun, ication]           2 tokens

The model might confuse "longer" (characters) with what it sees (tokens).
    By tokens: "extraordinarily" is longer  (3 vs 2) ✓
    By characters: "extraordinarily" = 15, "communication" = 13 ✓
    (Both agree here, but they don't always.)

Trickier:
    "Which is longer: 'strengths' or 'banana'?"
    "strengths" → [stre, ng, ths]     3 tokens, 9 characters
    "banana"    → [ban, ana]           2 tokens, 6 characters
    (Same direction. But the model sees 3 vs 2 tokens, not 9 vs 6 chars.)

Where it goes wrong:
    "Which is longer: 'queue' or 'qat'?"
    "queue" → [que, ue]     2 tokens, 5 characters
    "qat"   → [q, at]       2 tokens, 3 characters
    Same token count (2), different character count (5 vs 3).
    The model must go beyond token count to answer correctly.
```

---

## 6. Repeating a Word N Times

```text
"Write 'buffalo' exactly 7 times."

Seems trivial. But the model generates tokens autoregressively —
it doesn't have a counter. It predicts the next token based on
what came before, and "buffalo buffalo buffalo buffalo buffalo..."
all look the same to the pattern predictor.

Common failures:
    - Writes 6 or 8 instead of 7 (off-by-one)
    - Loses count for large N ("write 'test' 23 times")
    - For very large N, starts to drift or stop early

Why: the model has no internal counter variable.
It must track count through the context of its own output,
which becomes increasingly uniform ("buffalo" repeated).
Each new "buffalo" token is generated in nearly identical context.
```

---

## 7. Code Indentation and Whitespace

```text
Python is whitespace-sensitive. Tokenization makes this fragile.

"    x = 5"  (4 spaces) might tokenize as: [ĠĠĠĠ, x, Ġ=, Ġ5]
"   x = 5"   (3 spaces) might tokenize as: [ĠĠĠ, x, Ġ=, Ġ5]
"  x = 5"    (2 spaces) might tokenize as: [ĠĠ, x, Ġ=, Ġ5]

Three completely different first tokens for what looks almost identical.
The model must learn that:
    ĠĠĠĠ = one indentation level in some files
    ĠĠ   = one indentation level in other files
    Tab   = one indentation level in yet other files

Failure mode:
    Model generates code mixing 4-space and 2-space indentation.
    Looks fine visually, but Python throws IndentationError.
    The tokens for 4 spaces and 2 spaces are completely unrelated
    to the model — it doesn't "see" the space count, just token IDs.
```

---

## 8. Multilingual Token Tax

```text
Not just more tokens (file 03 covers this), but WORSE reasoning.

English:  "The cat sat on the mat"     → ~6 tokens
Korean:   "고양이가 매트 위에 앉았다"       → ~15 tokens
Japanese: "猫がマットの上に座った"         → ~12 tokens

Consequences beyond cost:
    1. The model has FEWER "thinking steps" per concept in non-English.
       If the model has 4096 output tokens, English gets ~700 words
       of reasoning. Korean gets ~250 words of equivalent reasoning.

    2. Each non-English token carries LESS meaning.
       English "cat" = one meaningful token.
       A Korean character might be a meaningless syllable fragment.
       The model must spend capacity reassembling meaning from fragments.

    3. Translating between languages can introduce errors at boundaries.
       "Translate: 人工知能" → the token boundaries in Chinese won't
       align with the token boundaries in "artificial intelligence."
       Cross-boundary information can get lost.

Practical impact:
    Models perform measurably worse on non-English benchmarks.
    Not (only) because they saw less non-English training data —
    also because tokenization forces them to work harder per concept.
```

---

## 9. Numbers and Dates

```text
"Is March 15, 2024 before or after April 2, 2024?"

Tokenized: [March, Ġ15, ,, Ġ2024, ...Ġ April, Ġ2, ,, Ġ2024]

The model doesn't see dates — it sees arbitrary token sequences.
To compare dates, it must:
    1. Understand "March" = month 3, "April" = month 4
    2. Compare: same year → compare months → March < April → before

This usually works for simple cases (learned from training data).
Failures appear with:
    - Edge cases: "Is Dec 31, 2023 before Jan 1, 2024?" (year boundary)
    - Relative dates: "What day is 45 days after Feb 14?"
      (requires knowing Feb has 28/29 days, then counting into March)
    - Unusual formats: "Is 2024-03-15 before 15/04/2024?"
      (different token patterns for the same concept)

Numbers have the same issue:
    "2024" and "2025" might have UNRELATED embeddings.
    The model doesn't know 2025 = 2024 + 1 from the token structure.
    It learned this from context during training.
```

---

## 10. Prompt Sensitivity (Same Meaning, Different Tokens)

```text
These three prompts mean the same thing but tokenize differently:

    "What's the capital of France?"
    "What is the capital of France?"
    " What is the capital of France?"   (leading space)

Tokenized differently:
    [What, 's, ...] vs [What, Ġis, ...] vs [ĠWhat, Ġis, ...]

Because the tokens differ, the attention patterns differ,
the internal representations differ, and the OUTPUT can differ.

Usually the answer is still "Paris" — but for harder tasks,
tiny prompt variations can cause different outputs:

    "Explain quantum computing"      → detailed technical answer
    "Explain quantum computing "     → (trailing space) might get
                                        a slightly different answer

    "Solve: 847 * 0.15"             → might get it right
    "Solve:  847 * 0.15"            → (double space) different tokens
                                       → might get it wrong

This is why prompt engineering matters (and is somewhat fragile).
The model doesn't process meaning — it processes tokens.
```

---

## The Pattern Behind All Failures

```text
Every failure in this file shares the same root cause:

    THE MODEL THINKS IN TOKENS, NOT IN [characters / digits /
    sounds / dates / concepts].

    When the task requires reasoning at a level FINER than tokens
    (characters, individual digits, phonemes), the model must:
        1. Decompose tokens into finer units (hard, error-prone)
        2. Reason about those units (possible, if step 1 worked)
        3. Recompose the answer into tokens (another error opportunity)

    When the task aligns WITH token granularity
    (word meanings, sentence structure, paragraph logic),
    models are excellent.

The two reliable workarounds:
    1. Chain-of-thought: re-encode the problem at finer granularity
       by spelling things out in the model's own output.
       Works for familiar patterns. Fragile for novel strings.

    2. Tool use: offload character/digit operations to Python or a
       calculator. 100% reliable. The model only needs to recognise
       WHEN to use a tool, not how to do the operation itself.

This is why modern AI systems (ChatGPT, Claude, etc.) are increasingly
tool-augmented rather than trying to do everything with raw generation.
The model reasons about WHAT to compute. The tool DOES the computation.
```
