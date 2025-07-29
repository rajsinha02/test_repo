import os
from typing import Callable, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import traceback

def process_image(image_path: str, output_path: str, analysis_fn: Callable, verbose: bool = True):
    try:
        df = analysis_fn(image_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        if verbose:
            print(f"Processed: {image_path}")
    except Exception as e:
        error_df = pd.DataFrame([{
            "raw_output": None,
            "processing_error": str(e),
            "traceback": traceback.format_exc(limit=2)
        }])
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        error_df.to_csv(output_path, index=False)
        if verbose:
            print(f"Failed: {image_path} — {e}")

def process_files(
    input_base: str,
    brands: List[str],
    output_base: str,
    analysis_fn: Callable,
    valid_extensions: List[str] = [".jpg", ".jpeg", ".png"],
    parallel: bool = True,
    verbose: bool = True
):
    tasks = []
    for brand in os.listdir(input_base):
        if brands and brand not in brands:
            continue

        brand_path = os.path.join(input_base, brand)
        if not os.path.isdir(brand_path):
            continue

        for video in os.listdir(brand_path):
            video_path = os.path.join(brand_path, video)
            if not os.path.isdir(video_path):
                continue

            for file in os.listdir(video_path):
                if file.lower().endswith(tuple(valid_extensions)):
                    file_path = os.path.join(video_path, file)
                    out_path = os.path.join(output_base, brand, video, f"{Path(file).stem}.csv")
                    tasks.append((file_path, out_path))

    print(f"Found {len(tasks)} files to process.")

    if parallel:
        with ThreadPoolExecutor() as executor:
            executor.map(lambda t: process_image(*t, analysis_fn, verbose), tasks)
    else:
        for task in tasks:
            process_image(*task, analysis_fn, verbose)

    print("All files processed.")
