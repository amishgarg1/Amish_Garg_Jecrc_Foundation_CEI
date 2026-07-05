import pypdf
import io
import os
from typing import List, Dict, Any, Union
from datasets import load_dataset

def extract_text_from_pdf(file_source: Union[str, bytes, io.BytesIO]) -> str:
    """
    Extracts unstructured raw text from a PDF file path, raw bytes, or BytesIO buffer.
    """
    if isinstance(file_source, str):
        # It's a file path
        with open(file_source, "rb") as f:
            reader = pypdf.PdfReader(f)
            text = "".join([page.extract_text() or "" for page in reader.pages])
    elif isinstance(file_source, (bytes, io.BytesIO)):
        # It's a binary buffer or bytes
        buffer = file_source if isinstance(file_source, io.BytesIO) else io.BytesIO(file_source)
        reader = pypdf.PdfReader(buffer)
        text = "".join([page.extract_text() or "" for page in reader.pages])
    else:
        raise ValueError("Unsupported file source type.")
    return text

def extract_text_from_txt(file_source: Union[str, bytes, io.BytesIO]) -> str:
    """
    Extracts text from a raw text file path, bytes, or BytesIO buffer.
    """
    if isinstance(file_source, str):
        with open(file_source, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif isinstance(file_source, bytes):
        return file_source.decode("utf-8", errors="ignore")
    elif isinstance(file_source, io.BytesIO):
        return file_source.getvalue().decode("utf-8", errors="ignore")
    else:
        raise ValueError("Unsupported file source type.")

def extract_text_via_gemini(file_bytes: bytes, api_key: str) -> str:
    """
    Extracts text from a PDF file (scanned or digital) using Gemini's multimodal capability.
    """
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    # Use gemini-1.5-flash which is fast and supports PDF input
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    prompt = (
        "Extract and transcribe all text from this document. "
        "Keep the formatting as close to the original as possible. "
        "Do not include any conversational filler, intro, or outro text. "
        "Return ONLY the transcribed text."
    )
    
    response = model.generate_content([
        {
            "mime_type": "application/pdf",
            "data": file_bytes
        },
        prompt
    ])
    return response.text.strip()

def load_hf_open_ragbench(dataset_name: str = "vectara/open_ragbench", split: str = "train", limit: int = 5) -> List[Dict[str, Any]]:
    """
    Loads samples from Hugging Face open_ragbench dataset using streaming mode.
    Returns a list of dictionaries containing keys: question, context, and ground_truth_answer.
    """
    print(f"Loading {limit} samples from HF dataset '{dataset_name}' (split: {split}) in streaming mode...")
    samples = []
    try:
        # Load dataset with streaming to save memory and avoid full download of large PDFs
        dataset = load_dataset(dataset_name, split=split, streaming=True)
        iterator = iter(dataset)
        
        for _ in range(limit):
            try:
                item = next(iterator)
                # Map the dataset columns. Typical open_ragbench items:
                # 'question', 'answer', 'context' (or 'text')
                
                # Check for standard fields
                question = item.get("question", "")
                answer = item.get("answer", "") or item.get("ground_truth", "") or item.get("target", "")
                
                # Context can be a list of passages or a single string
                context_raw = item.get("context", "") or item.get("text", "") or item.get("contexts", "")
                if isinstance(context_raw, list):
                    context = "\n\n".join([str(c) for c in context_raw])
                else:
                    context = str(context_raw)
                
                samples.append({
                    "id": item.get("id", len(samples)),
                    "question": question,
                    "context": context,
                    "answer": answer,
                    "raw_item": {k: v for k, v in item.items() if k not in ["context", "text", "images", "img_paths"]} # omit large objects
                })
            except StopIteration:
                break
    except Exception as e:
        print(f"Error streaming HF dataset: {e}")
        # Return empty list or basic mockup for offline development
    return samples

if __name__ == "__main__":
    # Quick test
    print("Ingestion module ready.")
