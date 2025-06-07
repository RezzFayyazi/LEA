# LEA
LLM Embedding-based Attribution (LEA): Quantifying Source Contributions to Generative Model's Response for Vulnerability Analysis

## Overview
This work proposes LLM Embedding-based Attribution (LEA) -- a novel, explainable metric to paint a clear picture on the `percentage of influence' the pre-trained knowledge vs. retrieved content has for each generated response. We apply LEA to verify that LLM-generated responses for vulnerability analysis are insightful, by showcasing the expected distribution of the responses when applied with verifiable sources, using 100 critical CVEs over a 10-year period.

![Alt text](images/dependency_process.png)
![Alt text](images/lea_table.png)

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
| `token_probs_diff_probs` | Plot Δ log‑probs between responses                               |


---

## 📊 Visualising Distributions

Generate LEA distributions by passing the path to the responses from all the LLMs (in the file):

```bash
$ python3 lea_distribution.py
```

---

## 📂 Repository Layout

```text
.
├── configs/           # YAML configs for generation & analysis
├── data/              # Curated CVE dataset (100 rows)
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
