from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Any

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    n: int = Field(1, ge=1)
    max_tokens: int = Field(500, ge=1)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    
class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class ChatCompletionResponseUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: ChatCompletionResponseUsage

class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]