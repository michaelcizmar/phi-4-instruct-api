import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from .api import router
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Phi-4 OpenAI-compatible API",
    description="FastAPI implementation of OpenAI-compatible API for Phi-4-mini-instruct model",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(router)

# Add health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

def start():
    """Start the API server."""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    uvicorn.run(
        "phi_4_instruct_api.main:app",
        host=host,
        port=port,
        reload=False,
        workers=1,
    )

if __name__ == "__main__":
    start()