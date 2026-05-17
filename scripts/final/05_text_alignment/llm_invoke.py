import instructor
from openai import OpenAI
from pydantic import BaseModel

client = instructor.from_openai(
    OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
    mode=instructor.Mode.JSON,
)

class PlagiarizedSpan(BaseModel):
    is_plagiarism: bool
    suspicious_start_char: int
    suspicious_end_char: int
    source_doc_id: str
    source_start_char: int
    source_end_char: int
    confidence: float
    reason: str

class AlignmentResult(BaseModel):
    spans: list[PlagiarizedSpan]