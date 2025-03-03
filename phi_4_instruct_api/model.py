import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List, Dict, Any, Optional
import logging
import time

logger = logging.getLogger(__name__)

class Phi4Model:
    def __init__(self, model_path: str = "microsoft/Phi-4-mini-instruct"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self.load_model()
        
    def load_model(self):
        """Load the model and tokenizer."""
        start_time = time.time()
        logger.info(f"Loading model from {self.model_path}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        
        logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    def generate(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int = 500,
        temperature: float = 0.0,
        top_p: float = 1.0,
        n: int = 1,
        stream: bool = False,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a response from the model."""
        start_time = time.time()
        
        generation_args = {
            "max_new_tokens": max_tokens,
            "return_full_text": False,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "top_p": top_p,
            **kwargs
        }
        
        # Handle streaming separately if implemented
        if stream:
            # This would be implemented with a generator
            raise NotImplementedError("Streaming is not yet implemented")
        
        # Generate response
        output = self.pipe(messages, **generation_args)
        response_text = output[0]['generated_text']
        
        # Apply stop sequences if provided
        if stop and isinstance(stop, list):
            for stop_seq in stop:
                if stop_seq in response_text:
                    response_text = response_text[:response_text.index(stop_seq)]
        
        generation_time = time.time() - start_time
        logger.info(f"Generated response in {generation_time:.2f} seconds")
        
        # Format response to match OpenAI API
        response = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_path,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": -1,  # Would need tokenizer to count
                "completion_tokens": -1,  # Would need tokenizer to count
                "total_tokens": -1  # Would need tokenizer to count
            }
        }
        
        return response

# Create a singleton instance
model_instance = None

def get_model_instance(model_path: str = "microsoft/Phi-4-mini-instruct"):
    """Get or create the model instance."""
    global model_instance
    if model_instance is None:
        model_instance = Phi4Model(model_path)
    return model_instance