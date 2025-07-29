
import os
import base64
import pandas as pd
from PIL import Image
import io
from pydantic import BaseModel
from typing import Optional, Tuple, Union
from llm_utils import init_llm, query_llm_with_image

# --- Schema Definitions ---
class ProductAndObjectDetection(BaseModel):
    recognizable_product: str
    percent_screen_occupied: Optional[str]
    percent_product_not_in_safe_zone: Optional[Union[float, str]]
    product_location_in_12_box_framework: Optional[int]

class ImageAnalysisOutput(BaseModel):
    product_and_object_detection: ProductAndObjectDetection

# --- Utility ---
def resize_and_encode(image_path: str, size: Tuple[int, int] = (800, 800)) -> str:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail(size)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode()

# --- Main Analysis Function ---
def analyze_image(image_path: str) -> pd.DataFrame:
    base64_img = resize_and_encode(image_path)
    prompt = """
You are an expert image analysis AI. Analyze this video frame describe the image
"""

    try:
        model = init_llm()
        structured = query_llm_with_image(base64_img, prompt, ImageAnalysisOutput, model)
        df = pd.json_normalize(structured.model_dump())
        df["processing_error"] = None
        return df
    except Exception as e:
        return pd.DataFrame([{"raw_output": None, "processing_error": str(e)}])
