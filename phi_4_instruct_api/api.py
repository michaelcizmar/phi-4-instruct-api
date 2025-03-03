from fastapi import APIRouter, HTTPException, Depends
from .schemas import ChatCompletionRequest, ChatCompletionResponse, ModelListResponse
from .model import get_model_instance, Phi4Model
import time
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    model: Phi4Model = Depends(get_model_instance)
):
    """Create a chat completion."""
    try:
        # Convert Pydantic messages to dict format expected by the model
        messages = [msg.dict() for msg in request.messages]
        
        # Generate response
        response = model.generate(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            n=request.n,
            stream=request.stream,
            stop=request.stop
        )
        
        return response
    except Exception as e:
        logger.exception("Error generating chat completion")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    """List available models."""
    model = get_model_instance()
    
    return {
        "object": "list",
        "data": [
            {
                "id": model.model_path,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "user"
            }
        ]
    }