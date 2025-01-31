"""
Medical Chroma DB Builder

This script processes medical symptom data using a language model and creates a ChromaDB vector database
for efficient similarity searches. The workflow includes data cleaning, LLM-powered processing, and 
persistent vector storage creation.

Environment Variables:
- OPENAI_API_KEY: OpenAI API key for embeddings
- HF_TOKEN: Hugging Face authentication token
- HF_MODEL: Hugging Face model identifier (default: ContactDoctor/Bio-Medical-Llama-3-8B)
"""

import os
import re
import json
import logging
from typing import Tuple, Dict, List

import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, logging as hf_logging
from datasets import load_dataset
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from huggingface_hub import login


# Configure logging
logging.basicConfig(from huggingface_hub import login
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress unnecessary warnings
hf_logging.set_verbosity_error()

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set")
login(HF_TOKEN, add_to_git_credential=True)
HF_MODEL = os.getenv("HF_MODEL", "ContactDoctor/Bio-Medical-Llama-3-8B")
PERSIST_DIR = os.getenv("PERSIST_DIR", "./chroma_db")

# System prompt template (maintained as a constant)
SYSTEM_PROMPT = """You are a medical expert that processes descriptions of disease symptoms. \
Return a JSON object with:
1. 'clean_description': A concise symptom summary (≤5 sentences)
2. 'symptoms': List of distinct symptoms (≤10 items)

Examples:
Input: [Polymorphous light eruption on the chest. Open pop-up dialog box. Close. Polymorphous light eruption on the chest. Polymorphous light eruption on the chest. Polymorphous light eruption is a rash that affects parts of the body that are exposed to more sunlight as daylight hours get longer, such as the front of the neck and chest. Polymorphous means that the rash can have many forms, such as tiny bumps, raised areas or blisters. Polymorphous light eruption on the chest. Open pop-up dialog box. Polymorphous light eruption on the chest. Open pop-up dialog box. Open pop-up dialog box . Close. Polymorphous light eruption on the chest. Polymorphous light eruption on the chest. Polymorphous light eruption is a rash that affects parts of the body that are exposed to more sunlight as daylight hours get longer, such as the front of the neck and chest. Polymorphous means that the rash can have many forms, such as tiny bumps, raised areas or blisters. Close. Polymorphous light eruption on the chest. Polymorphous light eruption on the chest. Polymorphous light eruption is a rash that affects parts of the body that are exposed to more sunlight as daylight hours get longer, such as the front of the neck and chest. Polymorphous means that the rash can have many forms, such as tiny bumps, raised areas or blisters. Close. Polymorphous light eruption on the chest. Polymorphous light eruption on the chest. Polymorphous light eruption is a rash that affects parts of the body that are exposed to more sunlight as daylight hours get longer, such as the front of the neck and chest. Polymorphous means that the rash can have many forms, such as tiny bumps, raised areas or blisters. Close. Close. Close Polymorphous light eruption on the chest . Polymorphous light eruption on the chest. Polymorphous light eruption is a rash that affects parts of the body that are exposed to more sunlight as daylight hours get longer, such as the front of the neck and chest. Polymorphous means that the rash can have many forms, such as tiny bumps, raised areas or blisters. Polymorphous light eruption on the chest Polymorphous light eruption is a rash that affects parts of the body that are exposed to more sunlight as daylight hours get longer, such as the front of the neck and chest. Polymorphous means that the rash can have many forms, such as tiny bumps, raised areas or blisters. Polymorphous light eruption is a rash that affects parts of the body that are exposed to more sunlight as daylight hours get longer, such as the front of the neck and chest. Polymorphous means that the rash can have many forms, such as tiny bumps, raised areas or blisters. How skin with sun allergy looks varies widely depending on the color of your skin and what's causing the symptoms. Signs and symptoms may include: Itchiness (pruritus). Stinging. Tiny bumps that may merge into raised patches. A flushing of the exposed area. Blisters or hives. Symptoms usually occur only on skin that has been exposed to the sun or other source of UV light. Symptoms show up within minutes to hours after sun exposure.]
Output: {"clean_description": "...", "symptoms": ["itchiness", ...]}

Input: [BPPV symptoms...]
Output: {"clean_descriptio": "Polymorphous light eruption is a rash that develops on parts of the body exposed to sunlight, like the chest and neck, particularly as daylight hours increase. The rash can appear in various forms, such as bumps, raised areas, or blisters. Symptoms may include itching, stinging, small bumps that merge into patches, and flushing of the skin, often appearing within minutes to hours of sun exposure. The rash primarily affects skin areas exposed to UV light and can vary depending on skin color and the cause of the reaction.", "symptoms": ['itchiness', 'stinging', 'small bumps that merge into patches', 'flushing of the skin']}"""


def configure_quantization() -> BitsAndBytesConfig:
    """
    Configure 4-bit quantization for model optimization.

    Returns:
        BitsAndBytesConfig: Quantization configuration object.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )


def initialize_llm() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Initialize the language model and tokenizer with proper configuration.

    Returns:
        Tuple: (tokenizer, model) pair.
    """

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL,
        device_map="auto",
        quantization_config=configure_quantization()
    )

    return tokenizer, model


def clean_source_text(text: str) -> str:
    """
    Clean and normalize medical text data.

    Args:
        text (str): Raw input text.

    Returns:
        str: Cleaned and normalized text.
    """
    patterns = [
        (r'\r|\t|\b', ''),          # Remove control characters
        (r'\n', '.'),               # Replace newlines with periods
        (r'[\\.]+', '. '),          # Normalize multiple periods
        (r':\s?\.', ':'),           # Fix incorrect punctuation
        (r'\s+', ' '),              # Collapse whitespace
        (r'(\.\s)+', '. '),         # Normalize period spacing
        (r'^\.', '')                # Remove leading periods
    ]

    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)

    return text.strip()


def process_medical_text(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM
) -> Dict[str, str]:
    """
    Process medical text through LLM to extract structured information.

    Args:
        text (str): Input medical text.
        tokenizer: Initialized tokenizer.
        model: Initialized language model.

    Returns:
        Dict: Processed output containing clean_description and symptoms.
    """
    user_prompt = f"Process the following medical text:\n{text}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(inputs, max_new_tokens=1000)
    response = tokenizer.decode(outputs[0]).split("Assistant: ")[-1].strip("<|eot_id|>")
    
    try:
        return json.loads(re.sub("'", '"', response))
    except json.JSONDecodeError:
        logger.error("Failed to parse model output: %s", response)
        return {"clean_description": "", "symptoms": []}


def enhance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhance medical dataset with cleaned descriptions and symptoms.

    Args:
        df (pd.DataFrame): Raw input DataFrame.

    Returns:
        pd.DataFrame: Enhanced DataFrame with new columns.
    """
    tokenizer, model = initialize_llm()
    
    df["text"] = df["text"].apply(clean_source_text)
    processed_data = df["text"].apply(
        lambda x: process_medical_text(x, tokenizer, model)
    )
    
    df["clean_description"] = processed_data.apply(lambda x: x.get("clean_description", ""))
    df["symptoms"] = processed_data.apply(
        lambda x: ", ".join(x.get("symptoms", [])) if x.get("symptoms") else ""
    )
    
    return df.dropna(subset=["symptoms"]).reset_index(drop=True)


def initialize_chroma_db(df: pd.DataFrame) -> Chroma:
    """
    Initialize or load ChromaDB vector store with medical data.

    Args:
        df (pd.DataFrame): Processed DataFrame.

    Returns:
        Chroma: Initialized ChromaDB instance.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    
    if os.path.exists(PERSIST_DIR):
        logger.info("Loading existing ChromaDB from %s", PERSIST_DIR)
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    
    logger.info("Creating new ChromaDB at %s", PERSIST_DIR)
    return Chroma.from_texts(
        texts=df["symptoms"].tolist(),
        metadatas=df[["label", "clean_description"]].to_dict("records"),
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )


"""Main execution flow for data processing and DB creation."""
try:
    logger.info("Loading source dataset")
    dataset = load_dataset("celikmus/mayo_clinic_symptoms_and_diseases_v1")["train"]
    df = pd.DataFrame(dataset, columns=["text", "label"])
    
    logger.info("Enhancing dataset with LLM processing")
    enhanced_df = enhance_dataset(df)
    
    logger.info("Initializing ChromaDB")
    chroma_db = initialize_chroma_db(enhanced_df)
    
    logger.info("Process completed successfully")
    print(f"Database contains {chroma_db._collection.count()} entries")
    
except Exception as e:
    logger.error("Processing failed: %s", str(e))
    raise