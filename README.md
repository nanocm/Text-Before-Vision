<div align="center">
  <h2><strong>Text Before Vision: Staged Knowledge Injection Matters for Agentic RLVR in UHR Remote Sensing</strong></h2>
  <h5>
  Anonymous Authors
      <br/><br/>
  </h5>
</div>

## 📚 Contents

- [📚Contents](#contents)
- [🔍Overview](#overview)
- [🌐Earth-Science Text QA Dataset](#earth-science-text-qa-dataset)
- [🛠️Methodology & Training](#methodology--training)
- [🚀Evaluation](#evaluation)

## 🔍Overview

![overview](assets/overview.jpg)

<p align="center"><strong>Fig 1. The impact of domain data and staged training on UHR Remote Sensing Understanding.</strong></p>

We introduce a novel **Staged Knowledge Injection** framework for Ultra-High-Resolution (UHR) Remote Sensing (RS) understanding. While Agentic Reinforcement Learning with Verifiable Rewards (RLVR) offers a path for navigating massive pixel spaces, we find that standard RL struggles without structured domain priors.

Our controlled studies yield a counter-intuitive finding: **High-quality Earth-science text-only QA is a primary driver of UHR visual reasoning gains.** Based on this, we propose a "Text-Before-Vision" recipe that achieves a **60.04% Pass@1 on XLRS-Bench**, establishing a new state-of-the-art.

The key contributions are:

* **Mechanistic Insights**: We demonstrate that the reasoning boundary (Pass@32) in UHR tasks is primarily governed by domain-prior coverage. Text-only QA instills reasoning structures that facilitate visual evidence retrieval.
* **Earth-Science Text QA Pipeline**: We release an automated pipeline and a dataset of **148,777 high-quality Text CoT QA pairs**, constructed from 8.8k textbooks and 200k scientific papers, rigorously verified by a domain-specific Knowledge Graph.
* **Staged Knowledge Injection Recipe**: We propose a training strategy: (1) **Cold-starting** with text QA to instill reasoning structures, followed by (2) **"Pre-warming"** on hard UHR image-text examples during SFT to stabilize subsequent tool-based Agentic RLVR.

## 🌐Earth-Science Text QA Dataset

We construct a large-scale, domain-specialized text QA dataset using a fully automated pipeline with "Active Pre-emptive Validation." The data generation process (Fig 2) utilizes a Knowledge Graph (built via LightRAG) to filter hallucinations before generation.

![pipeline](assets/pipeline.jpg)

<p align="center"><strong>Fig 2. Automated pipeline for Earth-science text QA generation and verification.</strong></p>

### Dataset Statistics

| Statistic | Value |
| :--- | :--- |
| **Total QA Pairs** | **148,777** |
| Avg. Question Length | 64.0 tokens |
| Avg. CoT + Answer Length | 256.9 tokens |
| Reasoning Steps (Avg) | 2.6 steps |
| Question Types | MCQ (24%), Fill (7%), T/F (4%), Free-form (65%) |

## 🛠️Methodology & Training

Our approach (Agentic RLVR) utilizes **Qwen2.5-VL** as the base model and integrates **GRPO** with zoom-in tools.

### 1. Prepare Data

* **Earth-Science Text QA**: Download our constructed dataset. 
* **SuperRS-VQA**: Ensure you have the SuperRS-VQA images for the visual warm-up stage.
* **General RL Data**: We utilize DeepEyes-47K for general reasoning stability.

**Anonymous Dataset URL**: https://huggingface.co/datasets/Anonymous-WR3Jazmla4/Text-Before-Vision

### 2. Training Stages

We recommend a staged training process to maximize Pass@1 and Pass@32 performance. Please check the `train` folder first.

#### Stage 1: Cold-Start SFT
We utilize LLaMA-Factory to perform SFT.

```bash
# please first install llamafactory, following
# https://github.com/hiyouga/LlamaFactory/tree/2a822178dea4d1c05f595521dd883a8e4f4e2e77
# download our data, modify dataset_info.json and file paths in yaml file
llamafactory-cli train yamls/base.yaml
# if encountered TypeError duiring dataset preprocess, refer to https://github.com/hiyouga/LlamaFactory/issues/5613
```

#### Stage 2: Agentic RLVR (with Tools)

Perform Group Relative Policy Optimization (GRPO) with zoom-in tools enabled. Here we utilize DeepEyes as the specific training framework.

```bash
# please first install DeepEyes, following
# https://github.com/Visual-Agent/DeepEyes
# download RL data, and modify parquet file paths in the training script
# follow deepeyes to set LLM judge and start training
bash train_rq_general.sh
```

## 🚀Evaluation

We evaluate primarily on **XLRS-Bench** to measure both average performance (Pass@1) and reasoning boundary (Pass@32).

### Evaluation

To replicate the results in Table 1 and Figure 4 of the paper:

```bash
# frist prepare data using the provided notebook
# convert model from pt format to hf model
bash s1.sh
# deploy model using vllm (or ray using `serve run ray.yaml`)
bash s21.sh
# prompting vllm, this may take 1~2 days considering different GPU types
bash s22.sh
# calculate metrics
bash s232.sh

```

### Expected Results

We also provide our trained model checkpoints through **anonymous** repo: https://huggingface.co/Anonymous-WR3Jazmla4/Text-Before-Vision

| Method | Pass@1 | Pass@32 |
| --- | --- | --- |
| Baseline (RLVR) | 50.01 | 82.58 |
| + Pre-warming (SuperRS-VQA) | 52.39 | 91.85 |
| **+ Text Cold Start (Ours)** | **60.40** | **96.25** |

