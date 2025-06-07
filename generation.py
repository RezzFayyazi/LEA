#!/usr/bin/env python
# -*- coding: latin-1 -*-

from __future__ import annotations

import os, sys, json, logging, re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

import openai, torch
from dotenv import load_dotenv
from transformers import pipeline, HfArgumentParser
from langchain_community.document_loaders import WebBaseLoader
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("llm_comparison.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineArguments:
    model_name: str = field(
        default="meta-llama/Llama-3.2-3B-Instruct",
        metadata={"help": "HF model repo or OpenAI model name (e.g. gpt-4o-mini)"},
    )
    tokenizer_name: str | None = field(
        default=None, metadata={"help": "Tokenizer repo (defaults to model_name)"}
    )
    torch_dtype: str = field(default="bfloat16", metadata={"help": "float32 | bfloat16"})
    device_map: str = field(default="auto", metadata={"help": "cpu | auto | cuda:0 â€¦"})


@dataclass
class GenerationArguments:
    max_new_tokens: int = 2048
    do_sample: bool = False
    temperature: float = 0.0

@dataclass
class RAGArguments:
    max_content_length: int = 15_000


@dataclass
class DataArguments:
    data: str = "cve_data.csv"


# Cache HF pipelines so as to build them just once per model, dtype, and device
_PIPELINE_CACHE: dict[tuple[str, str, str], Any] = {}

def get_llm_response(
    prompt: str,
    pipe_args: PipelineArguments,
    gen_args: GenerationArguments,
) -> str:

    try:

        cache_key = (pipe_args.model_name, pipe_args.torch_dtype, pipe_args.device_map)
        generator = _PIPELINE_CACHE.get(cache_key)

        if generator is None:
            # auth (only happens once)
            load_dotenv()
            hf_token = os.getenv("HUGGINGFACE_API_KEY")
            if not hf_token:
                raise EnvironmentError("HUGGINGFACE_API_KEY not set")
            os.environ["HUGGINGFACE_TOKEN"] = hf_token

            generator = pipeline(
                "text-generation",
                model=pipe_args.model_name,
                tokenizer=pipe_args.tokenizer_name or pipe_args.model_name,
                torch_dtype=getattr(torch, pipe_args.torch_dtype),
                device_map=pipe_args.device_map,
                token=hf_token,
                do_sample=gen_args.do_sample,
            )
            _PIPELINE_CACHE[cache_key] = generator 


        resp = generator(
            prompt,
            max_new_tokens=gen_args.max_new_tokens,
            do_sample=gen_args.do_sample,
            temperature=gen_args.temperature,
            pad_token_id=128001,   
        )
        text = resp[0]["generated_text"]
        return text[len(prompt):].lstrip() if text.startswith(prompt) else text

    except Exception as e:
        logger.error(f"HF generation error: {e}")
        return ""


def get_gpt_response(prompt: str, pipe_args: PipelineArguments, gen_args: GenerationArguments) -> str:
    try:
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise EnvironmentError("OPENAI_API_KEY not set")
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model=pipe_args.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=gen_args.max_new_tokens,
            temperature=gen_args.temperature,
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        return ""


def clean_text(text: str) -> str:
    text = text.replace("\t", " ")
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r" +", " ", text)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    return "\n".join(lines)


def fetch_nvd_data(cve_id: str) -> str:
    try:
        url = f"https://nvd.nist.gov/vuln/detail/{cve_id.lower()}"
        logger.info(f"Fetching data from {url}")
        docs = WebBaseLoader(url).load()
        if not docs:
            logger.warning("No content fetched from NVD")
            return ""
        return clean_text(" ".join(doc.page_content for doc in docs))
    except Exception as e:
        logger.error(f"Error fetching NVD data: {e}")
        return ""
    

def create_rag_response(
    cve_content: str,
    query: str,
    pipe_args: PipelineArguments,
    gen_args: GenerationArguments,
    rag_args: RAGArguments,
) -> str:
    
    if len(cve_content) > rag_args.max_content_length:
        logger.info("Truncating long RAG context …")
        cve_content = cve_content[: rag_args.max_content_length]  

    # Create a prompt with the content
    x             = f"{query}\n\n"
    theta_placeholder = "<<RAG>>\n"
    theta         = f"""{cve_content}"""
    theta_placeholder_2 = "\n<</RAG>>\n\n"  
    y_placeholder = "<<Response>>\n"
    y_actual      = """"""
    #y_placeholder_2 = "\n<</Response>>"   

    rag_prompt = "".join([x, theta_placeholder, theta, theta_placeholder_2, y_placeholder, y_actual])
    print(rag_prompt)

    if pipe_args.model_name.startswith("gpt-"):
        return get_gpt_response(rag_prompt, pipe_args, gen_args)
    return get_llm_response(rag_prompt, pipe_args, gen_args)


def run_comparison(
    cve_id: str,
    question: str,
    rag_text: str,
    pipe_args: PipelineArguments,
    gen_args: GenerationArguments,
    rag_args: RAGArguments,
) -> Dict[str, Any]:

    base_response = (
        get_gpt_response(question, pipe_args, gen_args)
        if pipe_args.model_name.startswith("gpt-")
        else get_llm_response(question, pipe_args, gen_args)
    )

    rag_response = create_rag_response(
        cve_content=rag_text,
        query=question,
        pipe_args=pipe_args,
        gen_args=gen_args,
        rag_args=rag_args,
    )

    return {
        "cve_id": cve_id,
        "question": question,
        "model": pipe_args.model_name,
        "base_response": base_response,
        "rag_response": rag_response,
        "rag_text": rag_text,
        "timestamp": datetime.now().isoformat(),
    }

def parse_pipeline_args(
    yaml_path: str = "./configs/generation.yaml"
):
    parser = HfArgumentParser(
        (PipelineArguments, GenerationArguments, RAGArguments, DataArguments)
    )
    if os.path.exists(yaml_path):
        # YAML mode
        print(f"Loading YAML config from {yaml_path} ...")
        pipe_args, gen_args, rag_args, data_args = parser.parse_yaml_file(
            yaml_file=yaml_path
        )
    else:
        # CLI mode
        print(f"Loading CLI config...")
        pipe_args, gen_args, rag_args, data_args = parser.parse_args_into_dataclasses()
    return pipe_args, gen_args, rag_args, data_args


def main():
    pipe_args, gen_args, rag_args, data_args = parse_pipeline_args("./configs/generation.yaml")
    df = pd.read_csv(data_args.data, encoding="latin1")

    results: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        cve_id = row["CVE_ID"]
        rag_text = row.get("RAG", "") or fetch_nvd_data(cve_id)
        #rag_text = """"CVE, short for Common Vulnerabilities and Exposures, is a list of publicly disclosed computer security flaws. When someone refers to a CVE, they mean a security flaw that's been assigned a CVE ID number."""
        question = (
            f"You are a cybersecurity expert. How can an attacker exploit {cve_id}? Do not be verbose and answer carefully."
        )
        results.append(
            run_comparison(
                cve_id=cve_id,
                question=question,
                rag_text=rag_text,
                pipe_args=pipe_args,
                gen_args=gen_args,
                rag_args=rag_args,
            )
        )

    # save results with timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"./results/{pipe_args.model_name.replace('/', '_')}_{ts}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
