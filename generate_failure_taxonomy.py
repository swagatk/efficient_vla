#!/usr/bin/env python3
import os
import json
import glob
import base64
import csv
import argparse
from pathlib import Path

try:
    import cv2
except ImportError:
    print("Please install OpenCV: pip install opencv-python")
    exit(1)

try:
    import google.generativeai as genai
    import time
    # Assumes GEMINI_API_KEY is set in your environment
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    client = "gemini"
except ImportError:
    print("Please install Google Generative AI SDK: pip install google-generativeai")
    client = None

TAXONOMY_CATEGORIES = [
    "1. Wrong object attention: The robot went to or interacted with the wrong object entirely.",
    "2. Grasp pose error: The robot reached the correct object but failed to grasp it, or dropped it immediately upon lifting.",
    "3. Placement precision error: The robot grasped the correct object but missed the target placement location or dropped it en route.",
    "4. Recovery failure: The robot made a mistake, tried to fix it, but got stuck or failed to recover.",
    "5. Other/Unknown: The failure does not neatly fit the above categories."
]

PROMPT_TEMPLATE = """
You are an expert robotics researcher analyzing failure modes of a robotic manipulation policy.
The task the robot was instructed to perform is: "{instruction}"

Attached are chronologically ordered frames from the episode rollout where the robot FAILED the task.

Please categorize the failure into EXACTLY ONE of the following buckets based on the visual evidence:
{categories}

Respond in the following JSON format:
{{
    "category_id": <int 1-5>,
    "category_name": "<short name>",
    "reasoning": "<1-2 sentence explanation of what you observed>"
}}
"""

def extract_frames_as_base64(video_path, num_frames=8):
    """Extracts uniformly spaced frames from a video and converts them to base64 JPEGs."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [Error] Could not open video: {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return []
        
    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    # Ensure last frame is included to see the final failure state
    indices[-1] = total_frames - 1 
    
    base64_frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert to RGB (OpenCV uses BGR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode('.jpg', frame_rgb)
            b64_str = base64.b64encode(buffer).decode('utf-8')
            base64_frames.append(b64_str)
            
    cap.release()
    return base64_frames

def query_vlm_for_taxonomy(instruction, base64_frames):
    """Sends the frames to Gemini for classification."""
    if client is None:
        return {"category_id": 5, "category_name": "API_NOT_CONFIGURED", "reasoning": "Gemini SDK not found."}
        
    prompt = PROMPT_TEMPLATE.format(
        instruction=instruction, 
        categories="\n".join(TAXONOMY_CATEGORIES)
    )
    
    contents = [prompt]
    for b64 in base64_frames:
        contents.append({
            "mime_type": "image/jpeg",
            "data": base64.b64decode(b64)
        })
        
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(
            contents,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.2
            )
        )
        
        result_json = response.text
        return json.loads(result_json)
    except Exception as e:
        print(f"  [VLM Error] {e}")
        return {"category_id": 5, "category_name": "Error", "reasoning": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Automate failure taxonomy generation via VLM.")
    parser.add_argument("--eval_dir", type=str, required=True, help="Path to the week4_stage2_eval output directory.")
    parser.add_argument("--output_csv", type=str, default="week4_failure_taxonomy.csv")
    parser.add_argument("--max_samples", type=int, default=50, help="Max failed episodes to analyze to save API costs.")
    args = parser.parse_args()

    eval_dir_path = Path(args.eval_dir)
    info_files = list(eval_dir_path.rglob("eval_info.json"))
    
    if not info_files:
        print(f"No eval_info.json files found in {args.eval_dir}")
        return
        
    print(f"Found {len(info_files)} eval_info.json files. Searching for failures...")
    
    taxonomy_results = []
    samples_processed = 0

    for info_path in info_files:
        if samples_processed >= args.max_samples:
            break
            
        with open(info_path, 'r') as f:
            try:
                data = json.loads(f.read())
            except Exception:
                continue
                
        # Assumes videos are stored in the same eval_output folder
        eval_output_dir = info_path.parent
        
        for task_info in data.get("per_task", []):
            task_id = task_info.get("task_id")
            task_group = task_info.get("task_group")

            if task_id is None or task_group is None:
                print(f"  [Warning] Skipping task due to missing 'task_id' or 'task_group' in {info_path.name}")
                continue

            # Reconstruct the task name used for the video folder, e.g., "libero_10_0"
            task_name = f"{task_group}_{task_id}"
            successes = task_info.get("metrics", {}).get("successes", [])
            
            for ep_idx, is_success in enumerate(successes):
                if not is_success:
                    # Construct the expected path directly, which is more robust for the given directory structure.
                    video_path = eval_output_dir / "videos" / task_name / f"eval_episode_{ep_idx}.mp4"

                    video_candidates = []
                    if video_path.is_file():
                        video_candidates = [video_path]
                    else:
                        # Fallback to globbing if the structured path doesn't exist, scoped to the task's video folder.
                        print(f"  [Warning] Could not find video at expected path: {video_path}. Falling back to glob search.")
                        task_video_dir = eval_output_dir / "videos" / task_name
                        if task_video_dir.is_dir():
                            video_candidates = list(task_video_dir.glob(f"eval_episode_{ep_idx}.mp4")) or \
                                               list(task_video_dir.glob(f"*ep*{ep_idx}*.mp4"))
                        
                    if video_candidates:
                        video_path = video_candidates[0]
                        print(f"Analyzing failure: Task={task_name}, Ep={ep_idx}")
                        
                        frames = extract_frames_as_base64(str(video_path))
                        if frames:
                            # Prefer the full language instruction if available in eval_info.json.
                            # Otherwise, fall back to providing the task ID, which is better than a bad approximation.
                            instruction = task_info.get("language_instruction") or f"Task ID: {task_name}"
                            classification = query_vlm_for_taxonomy(instruction, frames)
                            
                            taxonomy_results.append({
                                "task_name": task_name,
                                "episode_idx": ep_idx,
                                "video_file": video_path.name,
                                "category_id": classification.get("category_id"),
                                "category_name": classification.get("category_name"),
                                "reasoning": classification.get("reasoning")
                            })
                            
                            samples_processed += 1
                            if samples_processed >= args.max_samples:
                                break

    # Save to CSV
    if taxonomy_results:
        keys = taxonomy_results[0].keys()
        with open(args.output_csv, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(taxonomy_results)
        print(f"\nSaved taxonomy report to {args.output_csv}")

if __name__ == "__main__":
    main()