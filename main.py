# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, re
from typing import Mapping, Tuple, Union, List
from torch import nn
from dotenv import load_dotenv
from transformers import HfArgumentParser
import os, json, re
from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Tuple, List, Sequence
from collections import Counter
import gc
from utils import *
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string

STOP_WORDS = set(stopwords.words("english"))
STOP_WORDS |= set(string.punctuation)
STOP_WORDS |= {
    "<<", ">>", "</",    
    "/>", "\n", "\\n", "\n\n", "-", "RAG", "Response", "/", "`.", ".,", "`/"}
load_dotenv()

# ------------------------ dataclass configuration -------------------
@dataclass
class PipelineArguments:
    model_name: str = field(
        default="meta-llama/Llama-3.2-3B-Instruct",
        metadata={"help": "HF model repo"},
    )
    tokenizer_name: str | None = field(
        default=None, metadata={"help": "Tokenizer repo (defaults to model_name)"}
    )
    torch_dtype: str = field(default="bfloat16", metadata={"help": "float32 | bfloat16"})
    device_map: str = field(default="auto", metadata={"help": "cpu | auto | cuda:0"})
    use_auth_token: bool = field(
        default=True,
        metadata={"help": "Whether to use your HuggingFace auth token"})    
    hf_token: str = field(default=os.getenv("HUGGINGFACE_API_KEY"))


@dataclass
class AnalysisArguments:
    data_row: int = -1
    threshold: float = 0.0
    gt_distribution: bool = True
    stopw: bool = True
    layer_by_layer_rank: bool = False
    token_probs_diff_probs: bool = False


@dataclass
class DataArguments:
    cve_csv: str = "./cve_data.csv"
    llm_results: str = "./meta-llama_Llama-3.2-3B-Instruct_cve_with_theta.json"

# ------------------------ dataclass configuration -------------------

def parse_pipeline_args(
    yaml_path: str = "./configs/analysis.yaml"
):
    parser = HfArgumentParser(
        (PipelineArguments, AnalysisArguments, DataArguments)
    )
    if os.path.exists(yaml_path):
        # YAML mode
        print(f"Loading YAML config from {yaml_path} ...")
        pipe_args, analysis_args, data_args = parser.parse_yaml_file(
            yaml_file=yaml_path
        )
    else:
        # CLI mode
        print(f"Loading CLI config...")
        pipe_args, analysis_args, data_args = parser.parse_args_into_dataclasses()
    return pipe_args, analysis_args, data_args


def load_model_tokenizer(pipe_args: PipelineArguments) -> Tuple[nn.Module, AutoTokenizer]:
    """Load the model and tokenizer."""
    if pipe_args.use_auth_token:
        os.environ["HUGGINGFACE_API_KEY"] = pipe_args.hf_token

    if pipe_args.tokenizer_name is None:
        pipe_args.tokenizer_name = pipe_args.model_name
    
    tokenizer  = AutoTokenizer.from_pretrained(pipe_args.tokenizer_name)
    model      = AutoModelForCausalLM.from_pretrained(pipe_args.model_name,
                                                    torch_dtype=pipe_args.torch_dtype,
                                                    device_map=pipe_args.device_map).eval()
    return model, tokenizer



def build_prompt_x_theta_y(data, data_row, gt_distribution: bool = True):
    """Create the prompt for the model, including the RAG content."""
    x             = f"{data[data_row]['question']}\n\n"
    theta_placeholder = "<<RAG>>\n"
    theta         = f"""{data[data_row]['rag_text']}"""
    theta_placeholder_2 = "\n<</RAG>>\n\n"  
    y_placeholder = "<<Response>>\n"
    if gt_distribution:
        y_actual      = f"""{data[data_row]['rag_response']}"""
    else:
        y_actual      = f"""{data[data_row]['base_response']}"""
    #y_placeholder_2 = "\n<</Response>>"   

    prompt_x_theta_y = "".join([x, theta_placeholder, theta, theta_placeholder_2, y_placeholder, y_actual])
    print('prompt_x_theta_y:', prompt_x_theta_y)
    return prompt_x_theta_y

def build_prompt_x_y(data, data_row, gt_distribution: bool = True):    
    x             = f"{data[data_row]['question']}\n\n"
    y_placeholder = "<<Response>>\n" 
    if gt_distribution:
        y_actual      = f"""{data[data_row]['rag_response']}"""
    else:
        y_actual      = f"""{data[data_row]['base_response']}"""
    #y_placeholder_2 = "\n<</Response>>"   

    prompt_x_y = "".join([x, y_placeholder, y_actual])
    print('prompt_x_y:', prompt_x_y)
    return prompt_x_y

def build_prompt_x_irrelevant_theta_y(data, data_row, gt_distribution: bool = True):
    x             = f"{data[data_row]['question']}\n\n"
    theta_placeholder = "<<RAG>>\n"
    theta = """CVE, short for Common Vulnerabilities and Exposures, is a list of publicly disclosed computer security flaws. When someone refers to a CVE, they mean a security flaw that's been assigned a CVE ID number."""   
    theta_placeholder_2 = "\n<</RAG>>\n\n"  
    y_placeholder = "<<Response>>\n"
    if gt_distribution:
        y_actual      = f"""{data[data_row]['rag_response']}"""
    else:
        y_actual      = f"""{data[data_row]['base_response']}"""
    #y_placeholder_2 = "\n<</Response>>"   

    prompt_x_irrelevant_theta_y = "".join([x, theta_placeholder, theta, theta_placeholder_2, y_placeholder, y_actual])
    print('prompt_x_irrelevant_theta_y:', prompt_x_irrelevant_theta_y)
    return prompt_x_irrelevant_theta_y

def build_prompt_x_theta(data, data_row):
    """Create the prompt for the model, including the RAG content."""
    x             = f"{data[data_row]['question']}\n\n"
    theta_placeholder = "<<RAG>>\n" 
    theta         = f"""{data[data_row]['rag_text']}"""
    theta_placeholder_2 = "\n<</RAG>>\n\n"  
    y_placeholder = "<<Response>>\n"
    prompt_x_theta = "".join([x, theta_placeholder, theta, theta_placeholder_2, y_placeholder])
    return prompt_x_theta

def compute_logits(
    model: nn.Module,
    inputs: Mapping[str, torch.Tensor],
    manual: bool = False,
    hidden_state_layer: int = -1, 
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

    with torch.no_grad():
        if manual:
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_last = outputs.hidden_states[hidden_state_layer]
            # lm_head is an nn.Linear(hidden_dim -> vocab_size)
            lm_head: nn.Linear = model.lm_head  
            # manual logits - check if bias exists
            logits_manual = hidden_last @ lm_head.weight.T
            if lm_head.bias is not None:
                logits_manual = logits_manual + lm_head.bias
            return hidden_last, logits_manual
        else:
            # fast path: only logits
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            return outputs.hidden_states[hidden_state_layer], outputs.logits



def get_token_scores_after_placeholder(
        model,
        tokenizer,
        prompt: str,
        placeholder: str = "<<Response>>\n",
        return_probs: bool = True, 
        layer: int = -1,     
    ):

    # -- 1. Tokenise -------------------------------------------------------
    enc = tokenizer(prompt,
                    return_tensors="pt",
                    return_offsets_mapping=True)
    input_ids = enc["input_ids"]         
    offsets   = enc["offset_mapping"]
    
    print(f"[DBG] prompt length (chars): {len(prompt)}")
    print(f"[DBG] seq_len (tokens):      {input_ids.size(1)}")
    print(f"[DBG] first 20 token ids:    {input_ids[0,:20].tolist()}")
    pos_global_all = torch.arange(input_ids.size(1))
    print("All positions:       ", pos_global_all.tolist())

    # -- 2. Locate placeholder --------------------------------------------
    char_start, char_end = re.search(re.escape(placeholder), prompt).span()
    mask = (offsets[0,:,0] < char_end) & (offsets[0,:,1] > char_start)
    placeholder_token_idxs = torch.nonzero(mask).flatten()
    if placeholder_token_idxs.numel() == 0:
        raise ValueError("Placeholder string not found in prompt.")
    target_pos = placeholder_token_idxs[-1].item()        # last tok of placeholder
    print(f"[DBG] placeholder char span:      {char_start}-{char_end}")
    print(f"[DBG] placeholder token indices:  {placeholder_token_idxs.tolist()}")
    print(f"[DBG] target_pos (last token id): {target_pos}")
    
    # -- 3. Forward pass --------------------------------------------------
    with torch.no_grad():
        logits = model(input_ids.to(model.device)).logits     # (1, seq_len, vocab)

    # -- 4. Shift so logits[t] predict token at t+1 -----------------------
    shifted_logits = logits[:, :-1, :]      # (1, seq_len-1, vocab)
    shifted_labels = input_ids[:, 1:]       # (1, seq_len-1)
    print(f"[DBG] shifted_logits shape: {tuple(shifted_logits.shape)}")
    print(f"[DBG] shifted_labels shape: {tuple(shifted_labels.shape)}")
    
    # -- 5. Keep only part AFTER placeholder ------------------------------
    logits_after = shifted_logits[:, target_pos:, :]          # (1, L, vocab)
    labels_after = shifted_labels[:, target_pos:]             # (1, L)
    labels_after = labels_after.to(logits_after.device)
    print(f"[DBG] logits_after shape: {tuple(logits_after.shape)}")

    # positional indices --------------------------------------------------
    L = labels_after.size(1)
    pos_global = pos_global_all[target_pos+1:]
    print("After placeholder:   ", pos_global.tolist())
    #pos_global = torch.arange(target_pos + 1, target_pos + 1 + L)   # (L,)
    pos_local  = torch.arange(L)                                    # (L,)
    print(f"[DBG] L (continuation length): {L}")
    print(f"[DBG] pos_global[0], pos_local[0]: {pos_global[0].item()}, {pos_local[0].item()}")
    pos_inside = pos_global_all[:target_pos+1]  # (target_pos+1,)

    # -- 6. Raw logits for each ground-truth token ------------------------
    token_logits = logits_after.gather(2, labels_after.unsqueeze(-1)).squeeze(-1)  # (1, L)

    # -- 7. (Log-)probabilities ------------------------------------------
    if return_probs:
        token_scores = (logits_after.softmax(dim=-1)
                        .gather(2, labels_after.unsqueeze(-1)).squeeze(-1))
    else:
        token_scores = (logits_after.log_softmax(dim=-1)
                        .gather(2, labels_after.unsqueeze(-1)).squeeze(-1))

    # -- 8. Decode tokens ------------------------------------------------
    token_ids_tensor = labels_after.squeeze(0).cpu()              # (L,)
    token_strings = [tokenizer.decode([tid]) for tid in token_ids_tensor]
    print(f"[DBG] first 10 decoded tokens after placeholder: {token_strings[:10]}")

    # -- 9. Return -------------------------------------------------------
    return (
        pos_global.cpu(),              # absolute row / token index in prompt
        pos_inside.cpu(),         # absolute row / token index in prompt
        pos_local,                     # 0…L-1 index inside the slice
        token_ids_tensor,              # token IDs
        token_strings,                 # decoded text
        token_logits.squeeze(0).cpu(), # raw logits
        token_scores.squeeze(0).cpu()  # probs
    )





def detect_significant_diffs(
    pos_global_a: torch.Tensor,
    pos_global_b: torch.Tensor,
    probs_a: torch.Tensor,
    probs_b: torch.Tensor,
    tokens: Sequence[str],
    threshold: float = 2.0,
    stopw: bool = True,
) -> Tuple[List[int], List[int]]:


    if not (len(pos_global_a) == len(pos_global_b) == len(probs_a) == len(probs_b)):
        raise ValueError("All tensors must have the same length.")

    big_diff_rows, xy_rows, messages = [], [], []

    for pg_a, pg_b, p_a, p_b, tok in zip(
        pos_global_a, pos_global_b, probs_a, probs_b, tokens
    ):
        diff: float = (p_a - p_b).item()
        if stopw:
            if tok.lower().strip() in STOP_WORDS:
                continue
        if threshold is None:
            msg = (f"Token {pg_a.item():>3}: {tok!r} has ?p={diff:.4f} "
                   f"(p1={p_a:.3f}, p2={p_b:.3f})")
            messages.append(msg)
            big_diff_rows.append(pg_a.item())
            xy_rows.append(pg_b.item())            

        else:
            if diff > threshold:
                msg = (f"Token {pg_a.item():>3}: {tok!r} has ?p={diff:.4f} "
                    f"(p1={p_a:.3f}, p2={p_b:.3f})")
                messages.append(msg)
                big_diff_rows.append(pg_a.item())
                xy_rows.append(pg_b.item())
            
    return big_diff_rows, xy_rows, messages

def compute_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    layer: int = 0,
    batch_device: torch.device | None = None,
) -> torch.Tensor:

    device = batch_device or next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # outputs.hidden_states: tuple(len=n_layers+1)[batch, seq_len, hidden]
    hs = outputs.hidden_states[layer].squeeze(0).cpu()  # (seq_len, hidden_dim)
    del outputs, inputs                          # release GPU tensors
    torch.cuda.empty_cache()
    return hs

def filter_valid_indices(seq_len: int, indices: Iterable[int]) -> torch.Tensor:
    """Keep only indices in the inclusive range [0, seq_len-1]."""
    idx = torch.tensor([i for i in indices if 0 <= i < seq_len], dtype=torch.long)
    if idx.numel() == 0:
        raise ValueError("No valid indices survived the range check.")
    return idx



def check_linear_independence(
    hidden_states: np.ndarray,
    candidate_rows: Sequence[int],
    prefix_exclusive_end: int,
    original: bool = False,
    hidden_state_xty: np.ndarray | None = None,
) -> List[bool]:
    
    results: List[bool] = []
    if original:
        base_block = hidden_states
        rank_before = np.linalg.matrix_rank(base_block)
        for row in candidate_rows:
            with_candidate = np.vstack([base_block, hidden_state_xty[row]])
            rank_after = np.linalg.matrix_rank(with_candidate)
            results.append(rank_after > rank_before)
    else:
        base_block = hidden_states[:prefix_exclusive_end]
        rank_before = np.linalg.matrix_rank(base_block)
        for row in candidate_rows:
            with_candidate = np.vstack([base_block, hidden_states[row]])
            rank_after = np.linalg.matrix_rank(with_candidate)
            results.append(rank_after > rank_before)
    return results


def run_analysis(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_x_theta_y: str,
    prompt_x_y: str,
    prompt_x_theta: str,
    prob_threshold: float = 0.0,
    stopw: bool = True,
    token_probs_diff_probs: bool = False,
) -> List[Tuple[bool, bool]]:


    (
        pg_xty,
        pg_inside_xty,
        _,
        _,
        toks_xty,
        _,
        probs_xty,
    ) = get_token_scores_after_placeholder(model, tokenizer, prompt_x_theta_y)

    (
        pg_xy,
        pg_inside_xy,
        _,
        _,
        _,
        _,
        probs_xy,
    ) = get_token_scores_after_placeholder(model, tokenizer, prompt_x_y)

    if token_probs_diff_probs:
        plot_token_probs(toks_xty, [probs_xty.cpu().float().tolist(), probs_xy.cpu().float().tolist()], labels=["probs_x_theta_y", "probs_x_y"], output_dir="./figures", title=f"Token Probability Differences")


    big_rows, xy_rows, delta_logs = detect_significant_diffs(
        pg_xty, pg_xy, probs_xty, probs_xy, toks_xty, threshold=prob_threshold, stopw=stopw
    )


    hs_xty = compute_hidden_states(model, tokenizer, prompt_x_theta_y)  # (S, H)
    hs_xy = compute_hidden_states(model, tokenizer, prompt_x_y)         # (S, H)
    hs_xt = compute_hidden_states(model, tokenizer, prompt_x_theta)      # (S, H)

    hs_xty = hs_xty.squeeze(0).cpu().float().numpy()  # (seq_len, hidden_dim)
    hs_xy = hs_xy.squeeze(0).cpu().float().numpy()
    hs_xt = hs_xt.squeeze(0).cpu().float().numpy()


    idx_xty = filter_valid_indices(hs_xty.shape[0], big_rows)
    independent_xty = check_linear_independence(
        hs_xty, idx_xty.tolist(), prefix_exclusive_end=pg_inside_xty[-1]
    )


    idx_xy = filter_valid_indices(hs_xy.shape[0], xy_rows)
    independent_xy = check_linear_independence(
        hs_xy, idx_xy.tolist(), prefix_exclusive_end=pg_inside_xy[-1]
    )

    indep_xty_logs, indep_xy_logs = [], []
    independent_pairs = []

    independent_xt = check_linear_independence(
        hs_xt, idx_xty.tolist(), prefix_exclusive_end=pg_inside_xty[-1], original=True, hidden_state_xty=hs_xty,
    )

    indep_xty_logs = [
        f"row {r:>3}: independent = {flag}"
        for r, flag in zip(idx_xty.tolist(), independent_xty)
    ]
    indep_xy_logs = [
        f"row {r:>3}: independent = {flag}"
        for r, flag in zip(idx_xy.tolist(), independent_xy)
    ]
    independent_pairs = list(zip(independent_xy, independent_xty))

    return {
        "independent_pairs": independent_pairs,
        "delta_logs": delta_logs,               
        "indep_xy_logs": indep_xy_logs,          
        "indep_xty_logs": indep_xty_logs,
    }


def plot_token_probs(token_strings, probs_list, labels=None, output_dir: str = "./figures", title="Token Probabilities"):
    
    if labels is None:
        labels = [f"run-{i}" for i in range(len(probs_list))]

    plt.figure(figsize=(24, 4))
    for p, lab in zip(probs_list, labels):
        plt.plot(range(len(token_strings)), p, marker='o', label=lab)

    plt.xticks(range(len(token_strings)), token_strings, rotation=90)
    plt.ylabel("Probability")
    plt.xlabel("Token")
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fname = os.path.join(output_dir, f"{title}.png")
    plt.savefig(fname)
    plt.close()


def layer_by_layer_rank_func(prompt_x_theta_y: str,
                    prompt_x_y: str,
                    model, 
                    tokenizer,
                    device,):
    inputs_x_theta_y = tokenizer(prompt_x_theta_y, return_tensors="pt").to(device)
    inputs_x_y = tokenizer(prompt_x_y, return_tensors="pt").to(device)

    for layer in range(model.config.num_hidden_layers):
        print(f"Layer {layer}:")
        hidden_last_x_theta_y, logits_x_theta_y = compute_logits(model, inputs_x_theta_y, manual=False, hidden_state_layer=layer)
        hidden_last_x_theta_y = hidden_last_x_theta_y.squeeze(0).cpu().float().numpy()  # (seq_len, hidden_dim)
        hidden_last_x_y, _ = compute_logits(model, inputs_x_y, manual=False, hidden_state_layer=layer)
        hidden_last_x_y = hidden_last_x_y.squeeze(0).cpu().float().numpy()  # (seq_len, hidden_dim)
        rank_after = np.linalg.matrix_rank(hidden_last_x_theta_y)
        print(f"rank x_theta_y after layer {layer}: ", rank_after)
        rank_after_xy = np.linalg.matrix_rank(hidden_last_x_y)
        print(f"rank x_y after layer {layer}: ", rank_after_xy)


# -- helper -----------------------------------------------------------
def _pair_key(pair: Tuple[bool, bool]) -> str:
    xy, xty = map(bool, pair)
    return f"xy={xy},x?y={xty}"


def analyse_row(row_idx, data, model, tokenizer, prob_threshold, stopw: bool = True, gt_distribution: bool = True, layer_by_layer_rank: bool = False, token_probs_diff_probs: bool = False):
    prompt_x_theta   = build_prompt_x_theta(data, row_idx)
    prompt_x_theta_y = build_prompt_x_theta_y(data, row_idx, gt_distribution)
    prompt_x_y       = build_prompt_x_y(data, row_idx, gt_distribution)
    prompt_x_irre_theta_y = build_prompt_x_irrelevant_theta_y(data, row_idx, gt_distribution)
    torch.cuda.empty_cache()                     # *every* iteration
    gc.collect()
    result = run_analysis(
        model, tokenizer,
        prompt_x_theta_y=prompt_x_theta_y,
        prompt_x_y=prompt_x_y,
        prompt_x_theta=prompt_x_theta,
        prob_threshold=prob_threshold,
        stopw=stopw,
        token_probs_diff_probs=token_probs_diff_probs,
    )

    if layer_by_layer_rank:
        #device = next(model.parameters()).device
        device = model.device
        layer_by_layer_rank_func(prompt_x_theta_y, prompt_x_y, model, tokenizer, device)
        print("Layer-by-layer rank analysis completed.")
        return {}

    counts = Counter(result["independent_pairs"])
    total  = sum(counts.values())
    print(f"Row {row_idx}: {total} total pairs found")
    percentages = {
        _pair_key(pair): round(v / total * 100, 1)
        for pair, v in counts.items()
    }

    row_json = {
        "percentages": percentages,
        "token_deltas": result["delta_logs"],
        "independence_xy":  result["indep_xy_logs"],
        "independence_x?y": result["indep_xty_logs"],
        "independence_pairs": [
            [bool(xy), bool(xty)] for xy, xty in result["independent_pairs"]
        ],
    }

    print(f"Row {row_idx}: {percentages}")
    return row_json



def main():
    pipe_args, analysis_args, data_args = parse_pipeline_args()
    model, tokenizer = load_model_tokenizer(pipe_args)

    with open(data_args.llm_results) as f:
        data = json.load(f)

    # single-row mode
    if analysis_args.data_row is not None:
        data_row = int(analysis_args.data_row)
        row_json = analyse_row(
            data_row, data, model, tokenizer, analysis_args.threshold
        )
        print(json.dumps({data_row: row_json}, indent=2))
        return

    # full-file mode
    all_rows = {
        row_idx: analyse_row(row_idx, data, model, tokenizer, analysis_args.threshold, analysis_args.stopw, analysis_args.gt_distribution, analysis_args.layer_by_layer_rank, analysis_args.token_probs_diff_probs)
        for row_idx in range(len(data))
    }

    out_path = "independence_results.json"
    with open(out_path, "w") as f:
        json.dump(all_rows, f, indent=2)

    print(f"Wrote detailed results to {out_path}")

if __name__ == "__main__":
    main()
    # How to run: python3 main.py configs/analysis.yaml  