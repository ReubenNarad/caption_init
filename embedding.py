import os
import time
import numpy as np
from typing import List, Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

class QueryEmbedder:
    def __init__(self):
        # Initialize Gemini client
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Please set GEMINI_API_KEY environment variable")
        self.gemini_client = genai.Client(api_key=api_key)
        
    def embed_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Embed a batch of texts using Gemini"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                result = self.gemini_client.models.embed_content(
                    model="models/embedding-001",  # Using stable embedding model
                    contents=texts,
                    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
                )
                
                if result.embeddings:
                    return [np.array(emb.values) for emb in result.embeddings]
                else:
                    print(f"Empty embedding result for batch")
                    return [None] * len(texts)
                    
            except Exception as e:
                print(f"[Retry {attempt+1}/{max_retries}] Error embedding batch: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"Failed to embed batch after {max_retries} attempts")
                    return [None] * len(texts)
                