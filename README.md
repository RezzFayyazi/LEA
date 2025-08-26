# LEA
LLM Embedding-based Attribution (LEA): Quantifying Source Contributions to Generative Model's Response for Vulnerability Analysis

## Overview
This work proposes LLM Embedding-based Attribution (LEA), an explainable metric to quantify the influence of internal knowledge compared to retrieved content for model-generated responses. We evaluate LEA on 500 critical CVEs disclosed between 2016 and 2025 using three state-of-the-art LLMs. Our development of LEA exposes layer-wise dependence and reveals where context influence is strongest. 
We verify LEA's effectiveness in modeling the distribution of generated token dependence for vulnerability analysis. Our results demonstrate LEA’s ability to detect clear distinctions between no-retrieval, generic-retrieval, and ideal-retrieval scenarios. LEA offers security analysts with a metric to audit RAG-enhanced workflows, improving the transparent and trustworthy deployment of AI in cybersecurity threat analysis. 

<p align="center">
  <img src="images/dependency_process.PNG" alt="LEA end‑to‑end pipeline"/>
</p>


For each CVE, we considered three distinct retrieval scenarios:

Ideal retrieval: In this scenario, the LLM retrieves only the most relevant and verified information from the NVD website. This scenario serves as a benchmark to evaluate the LEA distribution under optimal retrieval conditions.

Generic retrieval: Here, we assume that the LLM does not have knowledge of the specific CVE and instead returns generalized information about CVEs, as illustrated below \cite{redhat}:

Failed retrieval: In this scenario, the LLM retrieves incorrect or misleading information, such as details from a non-existent CVE. For example, querying the model about CVE-2027-30066 (which does not exist) typically results in retrieval of CVEs sharing the same numeric suffix from other years.


## Setup
Create a virtual environment and install the libraries:

```sh
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

---

## How to Run

First, to run the huggingface models, create a `.env` file and put your API Key:

```env
HUGGINGFACE_API_KEY=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXX
```


### 🏗️ Generation Pipeline

`generation.py` produces model outputs **with retrieval** (RAG) **and without retrieval** (base) in a single run, storing them as JSON. To modify the generation arguments, go to `configs/generation.yaml` file and pass the desired arguments (such as the model's name). To run:

```bash
$ python3 generation.py configs/generation.yaml 
```




### 🧮 LEA Analysis

After generation, compute LEA scores and probability values with `main.py`:

```bash
$ python3 main.py configs/analysis.yaml
```

Essential knobs in **`configs/analysis.yaml`**:

| Field                    | Purpose                                                          |
| ------------------------ | ----------------------------------------------------             |
| `data_row`               | Process a single row (int) or *all* rows (omit)                  |
| `threshold`              | Filter tokens with attribution < τ                               |
| `gt_distribution`        | `True` → evaluate RAG response; `False` → evaluate base response |
| `layer_by_layer_rank`    | Print per‑layer rank statistics                                  |
| `token_probs_diff_probs` | Plot Δ softmax probs between responses                           |

---

## 📊 Visualising Distributions

Generate LEA distributions by passing the path to the responses for all the LLMs (in the file):

```bash
$ python3 lea_distribution.py
```

Generate the scatter plots for the $A^{rag}$ distributions across different RAG scenarios by passing the path to the responses for all the LLMs (in the file):

```bash
$ python3 plot_rag_dist.py
```


---

## 📂 Repository Layout

```text
.
├── configs/           # YAML configs for generation & analysis
├── data/              # Curated CVE dataset (500 rows)
│   └── cve_data.csv
├── results/
│   └── Generation/
│   └── LEA/           
├── images/            # Diagrams & figures used in the paper
└── *.py               # Entry points & core library
```

---


## 🖥️ Hardware & Performance

All experiments in the paper were run on a workstation with:

* **256 GB RAM**
* 2 × **Intel Xeon E5‑2650** CPUs
* 2 × **NVIDIA Tesla P40**
* 1 × **NVIDIA Tesla V100**
