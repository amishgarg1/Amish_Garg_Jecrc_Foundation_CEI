import re
from typing import List, Dict, Any

def extract_text_from_pdf(pdf_file) -> str:
    """
    Extracts text from an uploaded PDF file stream or file path.
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_file)
        text = ""
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {i+1} ---\n" + page_text
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

class RecursiveCharacterTextSplitter:
    """
    Splits text recursively by looking at different separators.
    """
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        return self._split(text, self.separators)

    def _split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]
        
        if not separators:
            # Fallback slice
            chunks = []
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunks.append(text[i : i + self.chunk_size])
            return chunks
        
        sep = separators[0]
        remaining_seps = separators[1:]
        
        if sep == "":
            splits = list(text)
        else:
            splits = text.split(sep)
            
        chunks = []
        current_chunk = []
        current_len = 0
        
        for part in splits:
            part_len = len(part)
            if part_len > self.chunk_size:
                if current_chunk:
                    chunks.append(sep.join(current_chunk))
                    current_chunk = []
                    current_len = 0
                sub_splits = self._split(part, remaining_seps)
                chunks.extend(sub_splits)
            elif current_len + part_len + (len(sep) if current_chunk else 0) <= self.chunk_size:
                current_chunk.append(part)
                current_len += part_len + (len(sep) if len(current_chunk) > 1 else 0)
            else:
                if current_chunk:
                    chunks.append(sep.join(current_chunk))
                # Add overlap
                overlap_chunk = []
                overlap_len = 0
                for item in reversed(current_chunk):
                    item_len = len(item)
                    if overlap_len + item_len + (len(sep) if overlap_chunk else 0) <= self.chunk_overlap:
                        overlap_chunk.insert(0, item)
                        overlap_len += item_len + (len(sep) if len(overlap_chunk) > 1 else 0)
                    else:
                        break
                current_chunk = overlap_chunk
                current_chunk.append(part)
                current_len = sum(len(x) for x in current_chunk) + len(sep) * (len(current_chunk) - 1)
                
        if current_chunk:
            chunks.append(sep.join(current_chunk))
            
        return [c for c in chunks if c.strip()]

def calculate_emi(principal: float, rate_pa: float, tenure_months: int) -> Dict[str, Any]:
    """
    Calculates the EMI and amortization summary.
    """
    if principal <= 0 or rate_pa <= 0 or tenure_months <= 0:
        return {"emi": 0.0, "total_payment": 0.0, "total_interest": 0.0, "schedule": []}
    
    r = (rate_pa / 12) / 100
    n = tenure_months
    
    # Formula: EMI = [P x r x (1+r)^n] / [(1+r)^n - 1]
    try:
        emi = (principal * r * ((1 + r) ** n)) / (((1 + r) ** n) - 1)
    except ZeroDivisionError:
        emi = principal / n
        
    total_payment = emi * n
    total_interest = total_payment - principal
    
    # Generate simple amortization schedule
    schedule = []
    balance = principal
    for month in range(1, n + 1):
        interest_payment = balance * r
        principal_payment = emi - interest_payment
        balance -= principal_payment
        balance = max(0.0, balance)
        schedule.append({
            "Month": month,
            "EMI": round(emi, 2),
            "Principal": round(principal_payment, 2),
            "Interest": round(interest_payment, 2),
            "Balance": round(balance, 2)
        })
        
    return {
        "emi": round(emi, 2),
        "total_payment": round(total_payment, 2),
        "total_interest": round(total_interest, 2),
        "schedule": schedule
    }

def check_eligibility(monthly_income: float, existing_emi: float, desired_loan: float, rate_pa: float, tenure_months: int) -> Dict[str, Any]:
    """
    Checks loan eligibility using Fixed Obligations to Income Ratio (FOIR).
    """
    # Define FOIR limit (50% for income < 50,000, 60% for income >= 50,000)
    foir = 0.50 if monthly_income < 50000 else 0.60
    
    max_emi_allowed = (monthly_income * foir) - existing_emi
    max_emi_allowed = max(0.0, max_emi_allowed)
    
    r = (rate_pa / 12) / 100
    n = tenure_months
    
    # Calculate required EMI for desired loan
    try:
        desired_emi = (desired_loan * r * ((1 + r) ** n)) / (((1 + r) ** n) - 1) if desired_loan > 0 else 0.0
    except ZeroDivisionError:
        desired_emi = desired_loan / n if n > 0 else 0.0
    
    # Calculate maximum eligible loan
    # P = [EMI * ((1+r)^n - 1)] / [r * (1+r)^n]
    if r > 0 and max_emi_allowed > 0:
        max_loan_eligible = (max_emi_allowed * (((1 + r) ** n) - 1)) / (r * ((1 + r) ** n))
    else:
        max_loan_eligible = max_emi_allowed * n
        
    eligible = desired_emi <= max_emi_allowed and desired_loan <= max_loan_eligible
    
    reason = "Salary meets requirements and overall EMIs are within limits." if eligible else ""
    if not eligible:
        if max_emi_allowed <= 0:
            reason = f"Existing EMIs exceed or equal the allowed FOIR limit (INR {monthly_income * foir:.2f}). No new EMI can be approved."
        else:
            reason = f"Desired EMI (INR {desired_emi:.2f}) exceeds maximum allowed monthly EMI limit (INR {max_emi_allowed:.2f})."
            
    return {
        "foir_percentage": foir * 100,
        "max_emi_allowed": round(max_emi_allowed, 2),
        "desired_emi": round(desired_emi, 2),
        "max_loan_eligible": round(max_loan_eligible, 2),
        "eligible": eligible,
        "status": "Approved" if eligible else "Refer / Not Approved",
        "reason": reason
    }
