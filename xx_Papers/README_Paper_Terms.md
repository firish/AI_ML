## Paper Terms & Abbreviations

A reference for understanding common terms, metrics, and abbreviations that appear in AI/ML research papers.

---

## Benchmarks & Datasets

| Term | Full Name | What It Is |
| --- | --- | --- |
| **WMT** | Workshop on Machine Translation | Annual competition for machine translation. "WMT 2014" = the 2014 edition. Papers report scores on WMT datasets to compare with other models. |
| **ImageNet** | — | Dataset of 1.3M labelled images across 1000 classes (cat, dog, car...). The standard benchmark for image classification. "ImageNet top-1 accuracy" = % of images correctly classified. |
| **MMLU** | Massive Multitask Language Understanding | 15,000+ multiple-choice questions across 57 subjects (math, history, law, medicine...). Tests how much an LLM "knows." Score is % correct. |
| **GPQA** | Graduate-Level Google-Proof QA | Very hard questions written by domain PhDs, designed so you can't just Google the answer. "Diamond" = the hardest subset. Score is % correct. |
| **HumanEval** | — | 164 Python programming problems. The model generates code, which is then executed against test cases. Score = "pass@1" = % of problems solved correctly on first try. |
| **MTEB** | Massive Text Embedding Benchmark | Leaderboard for text embedding models. Tests retrieval, classification, clustering, etc. across many datasets. |
| **GLUE / SuperGLUE** | General Language Understanding Evaluation | Collection of NLP tasks (sentiment, entailment, similarity). BERT-era benchmark, mostly saturated now. |
| **GSM8K** | Grade School Math 8K | 8,500 grade-school math word problems. Tests step-by-step reasoning. Score = % solved correctly. |
| **HellaSwag** | — | Sentence completion benchmark. Given a scenario, pick the most plausible continuation. Tests common sense. |
| **ARC** | AI2 Reasoning Challenge | Science exam questions (grade school level). "ARC-Challenge" = the harder subset. |
| **TriviaQA** | — | Trivia questions paired with evidence documents. Tests factual recall and reading comprehension. |
| **LAMBADA** | — | Predict the last word of a passage. Tests long-range language understanding. |
| **CommonCrawl** | — | Massive web scrape (~petabytes of text). Used as pre-training data for most LLMs. Not a benchmark — a data source. |
| **The Pile** | — | Curated 825GB dataset mixing books, code, Wikipedia, ArXiv, etc. Used for training open-source LLMs. |
| **LAION** | Large-scale Artificial Intelligence Open Network | Open datasets of image-text pairs. LAION-2B = 2 billion pairs. Used to train CLIP and image models. |
| **AudioSet** | — | 2M+ audio clips with labels (dog bark, music, speech...). Standard benchmark for audio classification. |

---

## Metrics

| Term | What It Measures | Scale | Example |
| --- | --- | --- | --- |
| **BLEU** | How close a machine translation is to a human reference translation. Measures n-gram overlap (1-gram, 2-gram, 3-gram, 4-gram matches). | 0-100, higher = better. 30+ is decent, 40+ is strong. | "28.4 BLEU" = the model's translations share ~28% of phrases with human translations. "+2 BLEU" is a large improvement. |
| **Perplexity (PPL)** | How "surprised" the model is by the next token. Lower = the model predicts better. Mathematically: `PPL = 2^(cross-entropy loss)`. | 1 to ∞, lower = better. | PPL = 4.92 means roughly "the model is choosing between ~5 equally likely options at each step." |
| **Top-1 Accuracy** | % of examples where the model's #1 prediction is correct. | 0-100% | "87.5% top-1 on ImageNet" = correctly classified 87.5% of images. |
| **Top-5 Accuracy** | % of examples where the correct answer is in the model's top 5 predictions. | 0-100% | Easier than top-1. Used when classes are ambiguous (is it a "dog" or "puppy"?). |
| **F1 Score** | Harmonic mean of precision and recall. Balances "did I find everything?" with "is what I found correct?" | 0-1 or 0-100 | F1 = 92.7 on parsing = the model correctly identifies 92.7% of parse tree structures. |
| **pass@k** | For code generation: run the model k times, did at least one attempt pass all test cases? | 0-100% | "pass@1 = 62.2%" = solves 62.2% of problems on the first try. |
| **ROUGE** | Measures overlap between generated summary and reference summary. Variants: ROUGE-1 (unigrams), ROUGE-2 (bigrams), ROUGE-L (longest common subsequence). | 0-1, higher = better | Used for summarisation tasks. |
| **FLOPs** | Floating Point Operations. Measures total compute used for training. | Raw number | 2.3 × 10¹⁹ FLOPs. Useful for comparing training cost across models. |
| **Latency** | Time to generate one response (or one token). | Milliseconds or seconds | "42ms per token" = time for each generated token. |
| **Throughput** | Tokens generated per second across all requests. | Tokens/sec | "10,000 tokens/sec" = serving capacity. |

---

## Model / Architecture Terms

| Term | What It Means |
| --- | --- |
| **Ensemble** | Running multiple models and combining their predictions (e.g., majority vote or average). Usually better than a single model but 3-5× more expensive. "Outperforms ensembles" = a single model beats multiple models combined — impressive. |
| **Backbone** | The main encoder network (e.g., ResNet-50, ViT-B). The "spine" of the model that extracts features. Other components (classification head, decoder) attach to it. |
| **Head** | A small layer on top of the backbone for a specific task. "Classification head" = linear layer → softmax. "Prediction head" = linear layer → vocabulary logits. |
| **Baseline** | The model you're comparing against. "Our method improves over the baseline by +2.0 BLEU" = we beat the reference model by 2 points. |
| **SOTA / SoTA** | State of the Art. The best known result on a benchmark. "New SOTA" = this model beat all previous models. |
| **Ablation** | Experiment where you remove or change one component to measure its importance. "Ablation study" = "what happens if we remove multi-head attention? What if we reduce dimensions?" |
| **Zero-shot** | The model performs a task it was never explicitly trained on. "Zero-shot classification" = classify images using text labels without any training examples of those labels. |
| **Few-shot** | Give the model a few examples in the prompt, then ask it to generalize. "5-shot" = 5 examples provided. |
| **Fine-tuning** | Continue training a pre-trained model on a smaller, task-specific dataset. |
| **Pre-training** | The initial large-scale training phase (on billions of tokens / millions of images). |
| **Inference** | Using the trained model to make predictions (no learning, weights frozen). |
| **Checkpoint** | A saved snapshot of model weights during training. "Averaged last 5 checkpoints" = averaged the weights from 5 different saves for a more stable model. |
| **Teacher forcing** | During training, feed the model the CORRECT previous token (not its own prediction). Faster training, but creates a mismatch with inference (where it uses its own predictions). |
| **Label smoothing** | Instead of training with hard targets [1, 0, 0], use soft targets [0.9, 0.05, 0.05]. Prevents overconfidence, improves generalization. |
| **Beam search** | Generation strategy: keep top-N candidates at each step instead of just the best one. "Beam size 4" = track 4 parallel hypotheses. More expensive but higher quality than greedy. |
| **Dropout** | Randomly zero out neurons during training to prevent overfitting (see NN Basics file 06). "P_drop = 0.1" = 10% of neurons dropped. |

---

## Scale & Compute Terms

| Term | What It Means | Example |
| --- | --- | --- |
| **B (params)** | Billion parameters | "7B model" = 7 billion trainable weights |
| **M (params)** | Million parameters | "65M model" = 65 million trainable weights |
| **T (tokens)** | Trillion tokens | "Trained on 1.4T tokens" = saw 1.4 trillion tokens during training |
| **P100 / A100 / H100** | NVIDIA GPU models | P100 (2016, 16GB), A100 (2020, 80GB), H100 (2022, 80GB). Each generation ~2-3× faster. |
| **GPU-hours** | Number of GPUs × hours of training | "8 GPUs × 84 hours = 672 GPU-hours" |
| **MoE** | Mixture of Experts | Only a subset of parameters activate per token. "48B total, 3B active" = 48B params exist but only 3B are used for each token. |
| **d_model** | Hidden dimension | Size of vectors throughout the model. 512, 768, or 4096 typically. |
| **d_ff** | Feed-forward inner dimension | Usually 4× d_model. The "expand then compress" in FFN blocks. |
| **h** | Number of attention heads | Usually 8, 12, or 32. Each head gets d_model/h dimensions. |
| **N** | Number of transformer layers/blocks | 6 (original), 12 (BERT/GPT-2), 32 (LLaMA-7B), 96 (GPT-3). |
| **n** | Sequence length | Number of tokens in the input. Context window limit. |
| **BPE** | Byte Pair Encoding | Tokenization method (see Phase 3, file 03). |
| **FP16 / BF16 / INT8 / INT4** | Number precision formats | FP16 = 16-bit float, INT4 = 4-bit integer. Lower precision = less memory, faster, slightly less accurate. |

---

## Common Paper Phrases

| Phrase | What it really means |
| --- | --- |
| "We achieve new state-of-the-art results" | Our model beats everything before it on this benchmark |
| "At a fraction of the training cost" | Our model is much cheaper to train |
| "We leave this to future work" | We didn't do this / it didn't work / we ran out of time |
| "Results are consistent across model sizes" | We tested on small, medium, and large — the trend holds |
| "Ablation studies confirm..." | We removed components one at a time to prove each one matters |
| "Our method is a drop-in replacement" | You can swap it into existing code without changing anything else |
| "With minimal overhead" | It costs slightly more compute, but the gains are worth it |
| "We observe diminishing returns" | It helps, but the improvement gets smaller the more you do it |
| "Outperforms even ensembles" | A single model beats multiple models averaged together — strong claim |
