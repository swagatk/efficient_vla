import cv2
import numpy as np
import torch
import os
from PIL import Image

try:
    from groundingdino.util.inference import Model
except ImportError:
    print("Warning: GroundingDINO not found. Please install groundingdino-py.")

class VisualPromptingWrapper:
    def __init__(self, use_image_box=True, use_text_hint=True, device="cuda"):
        self.use_image_box = use_image_box
        self.use_text_hint = use_text_hint
        self.device = device
        
        # 1. Initialize Grounding Model (GroundingDINO)
        # Point to the exact path in your GIT/efficient_vla folder where weights are downloaded
        base_dir = os.path.expanduser("~/GIT/efficient_vla")
        config_path = os.path.join(base_dir, "GroundingDINO_SwinT_OGC.py")
        weights_path = os.path.join(base_dir, "groundingdino_swint_ogc.pth")
        
        if os.path.exists(config_path) and os.path.exists(weights_path):
            try:
                self.model = Model(model_config_path=config_path, model_checkpoint_path=weights_path, device=self.device)
                print("GroundingDINO loaded successfully.")
            except Exception as e:
                print(f"Error loading GroundingDINO: {e}")
                self.model = None
        else:
            print(f"Warning: {config_path} or {weights_path} not found. Ensure weights are downloaded.")
            self.model = None

    def get_grounding_box(self, image_rgb: np.ndarray, text_instruction: str):
        """
        Extract the target object from the text instruction and query GroundingDINO for a bounding box.
        """
        if self.model is None:
            return None  # Fallback

        # Heuristic: extract the noun/target from the task instruction (e.g., "pick up the red block")
        target_object = self.extract_target_object(text_instruction)
        
        # GroundingDINO expects lower case text queries, separated by '.', and often benefits from suffixing '.'
        query = target_object.lower() + "."

        try:
            boxes, logits, phrases = self.model.predict_with_caption(
                image=image_rgb,
                caption=query,
                box_threshold=0.3,
                text_threshold=0.25
            )
            
            if len(boxes) == 0:
                return None # 3. Fallback mode: Grounding failed

            # Return the highest confidence box. 
            # Note: The output format from the Model wrapper is typically cxcywh in relative coords [0, 1].
            # We need to convert to absolute [x1, y1, x2, y2].
            best_idx = logits.argmax()
            h, w = image_rgb.shape[:2]
            best_box = boxes[best_idx].cpu().numpy()
            
            cx, cy, bw, bh = best_box
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            
            return [x1, y1, x2, y2]
            
        except Exception as e:
            print(f"GroundingDINO prediction error: {e}")
            return None

    def apply_prompts(self, observation: dict):
        """
        Intercepts the observation from the environment, modifies it, and returns the prompted observation.
        """
        img = observation['image'] # Expected (H, W, 3) numpy array
        instruction = observation['instruction']
        
        box = self.get_grounding_box(img, instruction)
        
        # 3. Fallback if grounding fails -> return original unmodified
        if box is None:
            return observation

        x1, y1, x2, y2 = map(int, box)
        
        # 2a. Prompt Encoding 1: Box-overlay channel on image
        if self.use_image_box:
            # Draw a thick prominent semi-transparent box or edge on the image
            prompted_img = img.copy()
            cv2.rectangle(prompted_img, (x1, y1), (x2, y2), (255, 0, 0), 3) # Red box
            
            observation['image'] = prompted_img

        # 2b. Prompt Encoding 2: Tokenized spatial hint in text prompt
        if self.use_text_hint:
            # Calculate object center relative to image dimensions
            h, w = img.shape[:2]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            
            # Simple grid tokens (e.g., top-left, center, bottom-right)
            x_loc = "left" if cx < w/3 else "center" if cx < 2*w/3 else "right"
            y_loc = "top" if cy < h/3 else "middle" if cy < 2*h/3 else "bottom"
            
            spatial_hint = f" [Target is located at {y_loc} {x_loc}]"
            observation['instruction'] = instruction + spatial_hint

        return observation

    def extract_target_object(self, instruction):
        """Simple heuristic to extract the main object from standard Libero instructions."""
        # E.g. "pick up the red block and put it in the bowl" -> "red block"
        instruction = instruction.lower()
        if "pick up the" in instruction:
            return instruction.split("pick up the")[1].split("and")[0].strip()
        return "object" # Fallback
