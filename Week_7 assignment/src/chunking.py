from typing import List, Dict, Any

class RecursiveCharacterTextSplitter:
    """
    A clean python implementation of recursive character-based text splitting.
    Attempts to split text recursively using a list of separators (e.g., paragraph, sentence, word)
    to keep logically coherent blocks of text together.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        return self._split_text(text, self.separators)

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        # If the text is already small enough, return it as a single chunk
        if len(text) <= self.chunk_size:
            return [text]

        # Select the separator to use
        separator = separators[-1] if separators else ""
        new_separators = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i+1:]
                break

        # Split text by the selected separator
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        # Merge splits back together to form chunks up to chunk_size with overlap
        chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split_len = len(split)
            # If a single split is larger than chunk_size, split it recursively
            if split_len > self.chunk_size:
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split the oversized segment recursively
                recursive_chunks = self._split_text(split, new_separators)
                chunks.extend(recursive_chunks)
            else:
                # Can we add this split to current_chunk?
                # Account for separator length
                sep_len = len(separator) if current_chunk else 0
                if current_length + sep_len + split_len > self.chunk_size:
                    # Current chunk is full, save it
                    chunks.append(separator.join(current_chunk))
                    
                    # Compute overlap: go backwards in splits to accumulate overlap
                    overlap_chunk = []
                    overlap_len = 0
                    for prev_split in reversed(current_chunk):
                        prev_sep_len = len(separator) if overlap_chunk else 0
                        if overlap_len + prev_sep_len + len(prev_split) <= self.chunk_overlap:
                            overlap_chunk.insert(0, prev_split)
                            overlap_len += prev_sep_len + len(prev_split)
                        else:
                            break
                    
                    current_chunk = overlap_chunk
                    current_length = overlap_len
                
                current_chunk.append(split)
                current_length += (len(separator) if len(current_chunk) > 1 else 0) + split_len

        if current_chunk:
            chunks.append(separator.join(current_chunk))

        # Filter out empty or whitespace-only chunks
        return [c.strip() for c in chunks if c.strip()]

def chunk_document(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Chunks a raw text document and returns a list of dictionaries with metadata.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    
    chunked_docs = []
    for idx, chunk in enumerate(chunks):
        chunked_docs.append({
            "chunk_id": idx,
            "text": chunk,
            "char_count": len(chunk),
            "word_count": len(chunk.split())
        })
    return chunked_docs

if __name__ == "__main__":
    test_text = "Hello world! This is a test. We want to see how chunking works. " * 20
    chunks = chunk_document(test_text, chunk_size=100, chunk_overlap=20)
    for c in chunks[:3]:
        print(c)
