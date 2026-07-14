# AI Engineering Lecture Notes

Detailed, structured notes distilled from long-form video lectures on building AI systems from first principles — starting at classic machine learning, working through how LLMs are actually built token by token, and ending with how to build a coding agent on top of one.

Each note set is a full write-up (table of contents, timestamps, diagrams, and worked examples) generated from a single multi-hour recorded lecture, so you can skim the outline or read section by section without rewatching the video.

## How to use this repo

Read the groups below in order — each one builds on the last. Click a title to open the notes.

---

## 1. Machine Learning Overview

The conceptual foundation before any code gets written: what ML actually is, how models learn, and the math underneath.

| Notes | What it covers |
|---|---|
| [Machine Learning Overview](machine-learning-overview/machine-learning-overview.md) | The AI → ML → DL → GenAI hierarchy, supervised/unsupervised/reinforcement learning, the training loop, core PyTorch/NumPy tooling, CUDA/MPS device management, and a map of the full ML lifecycle from features to deployment. |
| [Model Training & Optimisation — Part 1](model-training-and-optimisation-1/model-training-and-optimisation-1.md) | Gradient descent from scratch: weights, bias, linear regression, cost functions (MAE, RMSE), derivatives and slopes, learning rate tuning, backpropagation, and convergence/underfitting pitfalls. |

## 2. LLM From Scratch (Parts 1–5)

A five-part build-up of a GPT-style language model — starting from a 2-parameter toy model and ending with a fine-tuned, quantized, benchmarked transformer.

| Notes | What it covers |
|---|---|
| [LLM From Scratch — Part 1](llm-from-scratch-part-1/llm-from-scratch-part-1.md) | Course roadmap via real interview questions; tokenization, vocabulary, a dummy linear "LLM," PyTorch datasets/dataloaders, the training loop with AdamW, and autoregressive text generation. |
| [LLM From Scratch — Part 2](llm-from-scratch-part-2/llm-from-scratch-part-2.md) | Token + positional embeddings, building the model up from 2 → 30 → 78 parameters, softmax and logits, Jaccard similarity for word relatedness, and the shift toward self-attention. |
| [LLM From Scratch — Part 3](llm-from-scratch-part-3/llm-from-scratch-part-3.md) | Self-attention and scaled dot-product attention (Q/K/V) explained from RNN limitations upward, causal masking, residual connections, and assembling a full transformer block. |
| [LLM From Scratch — Part 4](llm-from-scratch-part-4/llm-from-scratch-part-4.md) | Saving/loading trained models, fine-tuning, GPT-2-style architecture (multi-head attention, dropout, layer norm), temperature/top-k/top-p sampling, and byte-pair encoding (BPE). |
| [LLM From Scratch — Part 5](llm-from-scratch-part-5/llm-from-scratch-part-5.md) | Evaluating models with perplexity, BPB, and LAMBADA accuracy; modern architecture upgrades (GQA, RMSNorm, RoPE); fine-tuning for tool calling; and LoRA for efficient adaptation. |

## 3. AI Coding Agent From Scratch (Parts 1–4)

Takes the LLM built above and wraps it in an agent loop — the same architecture behind tools like Claude Code — adding tools, safety, memory, and multi-agent orchestration.

| Notes | What it covers |
|---|---|
| [AI Coding Agent From Scratch — Part 1](ai-coding-agent-from-scratch-part-1/ai-coding-agent-from-scratch-part-1.md) | What an AI agent actually is, the core agent loop, LLM statelessness, a bash execution tool, safety/command-blocking, and context compaction for long sessions. |
| [AI Coding Agent From Scratch — Part 2](ai-coding-agent-from-scratch-part-2/ai-coding-agent-from-scratch-part-2.md) | Moving from raw bash to structured file tools (read/write schemas), sandboxing, system prompt design, manual context compaction, and a gatekeeper model for intent detection. |
| [AI Coding Agent From Scratch — Part 3](ai-coding-agent-from-scratch-part-3/ai-coding-agent-from-scratch-part-3.md) | Replacing a monolithic `CLAUDE.md` with dynamic, on-demand "Skills," introducing sub-agents as tools, context isolation, and running parallel sub-agents for token efficiency. |
| [AI Coding Agent From Scratch — Part 4](ai-coding-agent-from-scratch-part-4/ai-coding-agent-from-scratch-part-4.md) | Scaling to background/async sub-agents so the main loop stays responsive, multi-agent teams with a lead agent, workspace isolation via git worktrees, and streaming tool execution. |

## 4. Design a RAG System (Chat with Your Documents)

A full system-design walkthrough for a production Retrieval-Augmented Generation product, framed as an interview-style deep dive rather than a code build.

| Notes | What it covers |
|---|---|
| [Design a RAG System — Chat with Your Documents](design-a-rag-system-chat-with-your-documents/design-a-rag-system-chat-with-your-documents.md) | Why plain LLMs fail on private data, RAG vs. fine-tuning trade-offs, chunking/embedding at scale (1M+ chunks), hybrid retrieval (vector + BM25), reranking, vector DB choices (Qdrant/pgvector), and latency/cost budgets for a real production system. |

---

## Note format

Every notes file follows the same structure: a table of contents with timestamps back to the source lecture, then section-by-section write-ups with explanations, diagrams (Mermaid where useful), and code walkthroughs where the lecture writes code live.

## Contributing

These notes live on the `notes` branch and get merged into `main` via pull request as new lecture write-ups are added.
