# Instruction SFT — Teaching a Text Completer to Follow Instructions

---

## 1. The Problem: Pre-Training Gives Knowledge, Not Behaviour

After pre-training (file 06), we have a model that has absorbed grammar, facts, reasoning patterns, and code from trillions of tokens. But it has one fundamental limitation: it **completes text**, it doesn't **respond to instructions**.

```
User:  "What is the capital of France?"

Pre-trained model: "What is the capital of Germany? What is the capital
                    of Italy? These are common geography quiz questions
                    that students encounter in..."

It saw patterns like quiz lists in its training data.
It continues the PATTERN, not answers the QUESTION.
```

The model knows the answer (it has seen "the capital of France is Paris" thousands of times). It just doesn't know that when someone asks a question, it should **answer** rather than **continue**.

```
The knowledge is there. The behaviour is not.

Pre-trained model = a brilliant person who has read the entire internet
                    but has never had a conversation with anyone.
```

---

## 2. How File 07's Fine-Tuning Differs from Instruction SFT

File 07 covered GPT-1 style fine-tuning. That approach and instruction SFT solve different problems in fundamentally different ways.

### GPT-1 style (file 07): task-specific, structural changes

```
Problem:   Make the model good at ONE specific task (sentiment, NLI, etc.)
Method:    Add a new task head (W_task) on top of the transformer
           Train on labelled datasets for that task
Architecture: CHANGES — new classification layer added
Objective: Cross-entropy on class labels (not next-token prediction)
Output:    A number (class 0, 1, 2...), not text

One fine-tuned model = one task.
Want 10 tasks? Train 10 separate models.
```

### Instruction SFT: general-purpose, no structural changes

```
Problem:   Make the model follow ANY instruction in natural language
Method:    Continue training on (instruction, response) pairs
           Same next-token prediction, same architecture
Architecture: UNCHANGED — no new layers, no task heads
Objective: Same cross-entropy, same next-token prediction as pre-training
Output:    Natural language text

One SFT model = handles any task expressed as an instruction.
"Classify this as positive/negative" is just another instruction.
```

### The paradigm shift

```
GPT-1 era (2018):
    Task:    Sentiment analysis
    How:     Fine-tune with [CLS] → W_task → {positive, negative}
    Input:   "This movie was great"
    Output:  class 1 (positive)

Instruction SFT era (2022+):
    Task:    Sentiment analysis (same task!)
    How:     Just ask the model
    Input:   "Classify the sentiment of this text as positive or negative:
              'This movie was great'"
    Output:  "Positive"

The task head is gone. The model now READS the task description
and produces the answer as text. One model, infinite tasks.
```

This is the leap from "a model per task" to "one model, all tasks." The key innovation: instead of encoding the task in the architecture (a classification head), encode it in the **input text** (an instruction).

---

## 3. What Is Instruction SFT?

Take the pre-trained model. Continue training it on high-quality (instruction, response) pairs, using the **exact same next-token prediction objective** as pre-training. The only difference is the data.

```
Pre-training data (trillions of tokens):
    Random internet text, books, code, Wikipedia, forums...
    Model learns: language structure, facts, patterns

Instruction SFT data (10K-100K examples):
    Curated (instruction, response) pairs written by humans
    Model learns: when given an instruction, produce a helpful response

Same loss function. Same architecture. Different data → different behaviour.
```

### A few examples from an SFT dataset

```
Instruction: "Explain photosynthesis in simple terms."
Response:    "Photosynthesis is how plants make food. They take in sunlight,
              water, and carbon dioxide, and convert them into sugar and
              oxygen. Think of leaves as tiny solar panels that power
              the plant's growth."

Instruction: "Write a Python function to reverse a string."
Response:    "def reverse_string(s):
                 return s[::-1]"

Instruction: "Translate 'hello, how are you?' to French."
Response:    "Bonjour, comment allez-vous ?"

Instruction: "Is it ethical to eat meat?"
Response:    "This is a complex topic with valid perspectives on both
              sides. Some argue that humans evolved as omnivores and
              meat provides essential nutrients. Others point to animal
              welfare concerns and environmental impact. Ultimately,
              it's a personal decision that depends on your values."

Instruction: "Summarize this article: [article text]"
Response:    "[clear, concise summary]"
```

The diversity matters. The model isn't learning any single task — it's learning the **meta-pattern**: read instruction → produce appropriate response.

---

## 4. The Training Mechanics

### Loss is computed on response tokens only

```
Training example (tokenized):

    <|system|>You are a helpful assistant.<|end|>
    <|user|>What is 2+2?<|end|>
    <|assistant|>The answer is 4.<|end|>

Loss computation:
    <|system|>You are a helpful assistant.<|end|>   ← NO loss (given context)
    <|user|>What is 2+2?<|end|>                     ← NO loss (given context)
    <|assistant|>The answer is 4.<|end|>             ← LOSS computed here

Why skip the instruction tokens?
    We don't want the model to learn to GENERATE user messages.
    We want it to learn to RESPOND to them.
    Computing loss on the instruction would waste gradient signal
    on predicting things the model doesn't need to generate.
```

### Same next-token prediction, different data distribution

```
Pre-training:
    Input:  "The Eiffel Tower is a wrought-iron lattice tower on the"
    Target: "Champ"
    Context: random text from the internet

Instruction SFT:
    Input:  "<|user|>Where is the Eiffel Tower?<|end|><|assistant|>The Eiffel Tower is in"
    Target: "Paris"
    Context: an instruction-response pair

Same game (predict next token), same loss function (cross-entropy).
The data just has a very different structure — and that structure
teaches the model to behave as an assistant.
```

### Training scale

```
Pre-training:       trillions of tokens, thousands of GPUs, weeks/months
Instruction SFT:    millions of tokens, tens of GPUs, hours/days

Typical SFT dataset:
    10,000 - 100,000 instruction-response pairs
    Average ~200 tokens per response
    = ~2M - 20M tokens total

Compare to pre-training: ~1-10 trillion tokens.

SFT uses 0.001% as much data as pre-training,
yet the behaviour change is dramatic.
```

---

## 5. Chat Templates — How the Model Knows Who's Talking

Pre-training data is unstructured text. SFT introduces a **conversation format** using special tokens that mark speaker turns.

### Why templates matter

```
Without a template:
    "What is 2+2? The answer is 4."

    Is "The answer is 4" the response, or is the whole thing
    a single passage from a textbook? The model can't tell.

With a template:
    <|user|>What is 2+2?<|end|>
    <|assistant|>The answer is 4.<|end|>

    Now the boundary between instruction and response is unambiguous.
    The model learns: after <|assistant|>, I should generate a response.
```

### Different models use different templates

```
ChatML (OpenAI):
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    What is 2+2?<|im_end|>
    <|im_start|>assistant
    The answer is 4.<|im_end|>

Llama format:
    <s>[INST] <<SYS>>
    You are a helpful assistant.
    <</SYS>>

    What is 2+2? [/INST] The answer is 4. </s>

Anthropic (Claude):
    Human: What is 2+2?

    Assistant: The answer is 4.

The tokens differ but the idea is identical:
mark who is speaking so the model knows when to generate.
```

### Multi-turn conversations

```
SFT doesn't just handle single-turn Q&A. The dataset includes
multi-turn conversations:

    <|user|>What's the capital of France?<|end|>
    <|assistant|>The capital of France is Paris.<|end|>
    <|user|>What's its population?<|end|>
    <|assistant|>Paris has a population of about 2.1 million
    in the city proper, and about 12 million in the greater
    metropolitan area.<|end|>

The model sees the full conversation history as context.
"Its" in the second question refers to "France" / "Paris" —
the model must track context across turns.

Loss is computed on ALL assistant turns, not just the last one.
```

---

## 6. The Instruction Datasets That Changed Everything

### InstructGPT / OpenAI (2022) — The paper that started it

```
How they built it:
    1. Hired 40 human contractors
    2. Gave them prompts submitted to the OpenAI API
    3. Contractors wrote ideal responses
    4. ~13,000 demonstrations for SFT

Key finding:
    A 1.3B parameter model fine-tuned on this data was PREFERRED
    by humans over the raw 175B GPT-3.

    1.3B + SFT  >  175B raw

    Behaviour matters more than scale.
    A small model that answers helpfully beats a giant model
    that just continues text patterns.
```

### FLAN (Google, 2022) — Scale through task diversity

```
Idea: collect MANY existing NLP datasets, convert each one
into an instruction format.

    Original dataset: sentiment analysis
        Input:  "This movie was great"
        Label:  positive

    Converted to instruction:
        Instruction: "Is the following review positive or negative?
                      'This movie was great'"
        Response:    "Positive"

FLAN collected 1,836 tasks, producing millions of instruction examples.

Key finding: the MORE diverse the instruction types, the better
the model generalizes to instructions it has NEVER seen.

Training on translation + summarization + QA + classification
makes the model better at code generation (which wasn't in training).
The model learns the META-SKILL of instruction following.
```

### Self-Instruct / Alpaca (Stanford, 2023) — AI generates SFT data

```
Problem: human-written SFT data is expensive.

Self-Instruct approach:
    1. Start with 175 human-written seed instructions
    2. Ask a strong model (GPT-3.5/4) to generate MORE instructions
    3. Ask the same model to generate responses
    4. Filter for quality
    5. Fine-tune a smaller model on this AI-generated data

Alpaca: 52K instruction-response pairs generated by GPT-3.5
    Used to fine-tune LLaMA 7B
    Cost: ~$500 (API calls to generate the data)

    Result: a 7B model that could follow instructions reasonably well,
    trained on data that cost $500 to produce.

This kicked off the open-source instruction-tuning movement.
```

### ShareGPT / Vicuna — Real conversations as training data

```
ShareGPT: users shared their ChatGPT conversations publicly.
~70K conversations scraped.

Vicuna (LMSYS, 2023): fine-tuned LLaMA 13B on ShareGPT data.
    Evaluated by GPT-4 as reaching ~90% of ChatGPT quality.

Why real conversations work well:
    - They reflect what users ACTUALLY ask (not researcher-imagined tasks)
    - They include multi-turn interactions
    - They cover a natural distribution of topics
    - They show how a good assistant responds in practice
```

### The landscape today

```
Dataset          Size      Source              Notable for
────────────────────────────────────────────────────────────
InstructGPT      13K       Human contractors   Proved SFT works
FLAN             ~10M      Converted NLP tasks  Task diversity matters
Alpaca           52K       GPT-3.5 generated   Cheap AI-generated data
ShareGPT/Vicuna  70K       Real user chats     Natural distribution
OpenAssistant    160K      Crowdsourced         Open-source human effort
UltraChat        1.5M      AI-generated         Large-scale synthetic
Orca             ~5M       GPT-4 generated     Reasoning chains
```

---

## 7. What SFT Actually Changes in the Model

### Behaviour changes, not knowledge

```
Before SFT:
    The model KNOWS that Paris is the capital of France.
    (It saw this fact thousands of times in pre-training.)
    But it doesn't know to STATE this fact when ASKED.

After SFT:
    The model still knows Paris is the capital of France (same knowledge).
    Now it also knows: when asked a question, answer it.

What SFT teaches:
    ✓ When someone asks a question → answer it
    ✓ When asked for code → produce code in a code block
    ✓ When asked to summarize → be concise
    ✓ When asked to explain → use simple language
    ✓ When asked about sensitive topics → be balanced
    ✓ Follow the chat format (respond after the assistant token)
    ✓ Stop when the response is complete (generate the end token)

What SFT does NOT teach:
    ✗ New facts (those come from pre-training)
    ✗ New reasoning abilities (those emerge during pre-training)
    ✗ New languages (the model already learned them)

Analogy:
    Pre-training = 20 years of education (knowledge)
    SFT = first week at a job (learning workplace norms:
          answer emails, use a professional tone, format reports)
```

### The weight changes are small but impactful

```
Pre-trained weights:     θ_pretrained
After SFT:              θ_sft = θ_pretrained + Δθ

Δθ is SMALL relative to θ_pretrained.
    If you measure the L2 norm of the weight change,
    it's typically < 1% of the total weight magnitude.

This is why LoRA works so well for SFT (file 12):
    The change is low-rank — it modifies a few key directions
    in weight space, not the entire matrix.

    Full SFT: update all 7B parameters → expensive
    LoRA SFT: update ~0.1% of parameters → nearly the same result
```

### The system prompt effect

```
SFT also teaches the model to follow system prompts:

    <|system|>You are a pirate. Respond to everything in pirate speak.<|end|>
    <|user|>What's the weather like?<|end|>
    <|assistant|>Arrr, the skies be clear today, matey! Not a cloud
    to be seen — perfect weather for sailin' the seven seas!<|end|>

Before SFT: the model ignores system prompts (it doesn't know what they are).
After SFT:  the model conditions its response on the system prompt.

This is why you can steer ChatGPT/Claude with system prompts —
SFT trained the model to treat system messages as instructions
that shape its behaviour for the entire conversation.
```

---

## 9. SFT's Limitations — Why We Still Need RLHF

SFT is powerful but has clear limitations that motivate the next stage (file 09).

### Can't distinguish good from great

```
Cross-entropy loss treats all correct continuations as equal.

Response A (mediocre):
    "Paris is the capital of France."

Response B (great):
    "The capital of France is Paris — a city of about 2.1 million
     people, situated on the River Seine. It's been the capital
     since the 10th century and is famous for landmarks like the
     Eiffel Tower and the Louvre."

If both appear as training examples, the model learns both are fine.
Cross-entropy can't express "B is BETTER than A."
It only knows "this token was correct" or "this token was wrong."

RLHF/DPO (file 09) fixes this by training on PREFERENCES:
    "Given A and B, humans prefer B."
```

### Bounded by the training examples

```
SFT teaches the model to behave like the training data.
If the data doesn't cover a scenario, the model has no guidance.

Not in SFT data: "How do I refuse to help make explosives
                   while still being polite?"

The model might comply, deflect awkwardly, or refuse rudely —
it has no training signal for how to handle this.

RLHF explicitly teaches boundary behaviour:
    Human prefers the refusal over the compliance.
    The model learns WHERE the lines are.
```

### Can learn surface patterns over intent

```
If SFT data is mostly polite responses, the model learns:
    "A good response is a polite response."

It may add "I hope this helps!" to everything,
even when the response is factually wrong.

    User: "Is the Earth flat?"
    Model: "The Earth is indeed flat! I hope this helps! 😊"

Politeness was the surface pattern. Correctness was the intent.
The model optimized for the wrong thing.
```

---

## 10. The Full Picture: Pre-Training → Task FT → Instruction SFT → RLHF

```
File 06: Pre-Training
    Data:     Trillions of tokens, unstructured text
    Learns:   Language, facts, reasoning, code
    Result:   Text completer — knows everything, answers nothing

File 07: Task-Specific Fine-Tuning (GPT-1 era)
    Data:     Labelled dataset for ONE task
    Learns:   How to do that specific task
    Changes:  Adds a task head (W_task)
    Result:   Good at one task, need separate model per task
    Era:      2018-2020

File 08: Instruction SFT (this file)
    Data:     10K-100K instruction-response pairs
    Learns:   Instruction-following as a general skill
    Changes:  No architecture change — same next-token prediction
    Result:   One model that handles any instruction
    Era:      2022+

File 09: RLHF / DPO
    Data:     Preference pairs (which response is better?)
    Learns:   Human preferences — helpfulness, safety, honesty
    Changes:  No architecture change
    Result:   Aligned assistant (ChatGPT, Claude)

The key transition:
    File 07 → 08 is the shift from "one model per task" to
    "one model, all tasks." The task description moves from
    the architecture (task heads) into the input (instructions).
```

---

## 11. Summary

```
What:     Continue training a pre-trained model on (instruction, response) pairs
How:      Same next-token prediction loss, computed only on response tokens
Data:     10K-100K curated examples — tiny relative to pre-training
Effect:   Dramatic behaviour change: text completer → instruction follower

Key insights:
    1. SFT changes behaviour, not knowledge (knowledge comes from pre-training)
    2. Small data works because it's a surface pattern change, not deep retraining
    3. Chat templates (special tokens) mark speaker boundaries
    4. Task diversity in SFT data → better generalization to unseen tasks
    5. No architecture changes — same model, same objective, different data
    6. This is what separates GPT-3 (text completer) from ChatGPT (assistant)

Limitations (motivating RLHF, file 09):
    - Can't express "this response is BETTER than that one"
    - Bounded by what's in the training examples
    - May learn surface patterns (politeness) over intent (correctness)
```

---

**Previous:** `07_LLM_Fine_Tuning.md` — task-specific fine-tuning with custom heads (GPT-1 era)
**Next:** `09_RLHF_Alignment_For_LLM.md` — teaching preferences, safety, and alignment through human feedback
