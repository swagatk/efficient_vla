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

    def get_grounding_box_for_target(self, image_rgb: np.ndarray, target: str):
        """
        Query GroundingDINO for a bounding box of a specific target object.
        """
        if self.model is None:
            return None  # Fallback

        query = target.lower() + "."

        try:
            outputs = self.model.predict_with_caption(
                image=image_rgb,
                caption=query,
                box_threshold=0.3,
                text_threshold=0.25
            )
            
            if len(outputs) == 2:
                detections, phrases = outputs
                if len(detections) == 0:
                    return None
                best_idx = detections.confidence.argmax()
                x1, y1, x2, y2 = detections.xyxy[best_idx]
            else:
                boxes, logits, phrases = outputs
                if len(boxes) == 0:
                    return None # 3. Fallback mode: Grounding failed

                # Return the highest confidence box. 
                # Note: The output format from the Model wrapper is typically cxcywh in relative coords [0, 1].
                # We need to convert to absolute [x1, y1, x2, y2].
                best_idx = logits.argmax()
                h, w = image_rgb.shape[:2]
                best_box = boxes[best_idx].cpu().numpy() if hasattr(boxes, 'cpu') else boxes[best_idx]
                
                cx, cy, bw, bh = best_box
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
            
            return [int(x1), int(y1), int(x2), int(y2)]
            
        except Exception as e:
            print(f"GroundingDINO prediction error: {e}")
            return None

    def apply_prompts(self, observation: dict):
        """
        Intercepts the observation from the environment, modifies it, and returns the prompted observation.
        """
        img = observation['image'] # Expected (H, W, 3) numpy array
        instruction = observation['instruction']
        
        targets = self.extract_target_objects(instruction)
        
        prompted_img = img.copy() if self.use_image_box else img
        new_instruction = instruction
        
        for query_target, replace_target in targets:
            box = self.get_grounding_box_for_target(img, query_target)
            if box is None:
                continue
                
            x1, y1, x2, y2 = map(int, box)
            
            if self.use_image_box:
                cv2.rectangle(prompted_img, (x1, y1), (x2, y2), (255, 0, 0), 3) # Red box
                
            if self.use_text_hint:
                # Provide the exact bounding box coordinates instead of a coarse 3x3 grid
                hint_str = f"[{replace_target}, box: {x1} {y1} {x2} {y2}]"
                # Replace the exact occurrence of the target in the instruction
                new_instruction = new_instruction.replace(replace_target, hint_str, 1)

        if self.use_image_box:
            observation['image'] = prompted_img
            
        if self.use_text_hint:
            observation['instruction'] = new_instruction

        return observation

    def extract_target_objects(self, instruction):
        """
        Extract multiple objects from Libero instructions.
        Returns a list of tuples: (query_for_dino, string_to_replace_in_instruction)
        """
        instruction = instruction.lower()
        targets = []
        
        # Helper to strip spatial prepositional phrases
        def clean_obj(obj_str):
            for separator in [" on ", " in ", " to ", " next to ", " from ", " into ", " inside "]:
                obj_str = obj_str.split(separator)[0]
            return obj_str.strip()

        if "put both the" in instruction:
            objects_str = clean_obj(instruction.split("put both the ")[1])
            for t in objects_str.split(" and the "):
                t = clean_obj(t)
                targets.append((t, t))
        elif "put both" in instruction:
            objects_str = clean_obj(instruction.split("put both ")[1])
            if objects_str.endswith("s"):
                targets.append((objects_str[:-1], objects_str))
            else:
                targets.append((objects_str, objects_str))
        elif " and put the " in instruction:
            parts = instruction.split(" and put the ")
            t1 = clean_obj(parts[0].replace("put the ", "").replace("turn on the ", ""))
            t2 = clean_obj(parts[1])
            targets.append((t1, t1))
            targets.append((t2, t2))
        else:
            if "pick up the" in instruction:
                t = clean_obj(instruction.split("pick up the")[1].split(" and ")[0])
                targets.append((t, t))
            elif "put the" in instruction:
                t = clean_obj(instruction.split("put the")[1])
                targets.append((t, t))
            elif "turn on the" in instruction:
                t = clean_obj(instruction.split("turn on the")[1].split(" and ")[0])
                targets.append((t, t))
            else:
                targets.append(("object", "object"))
                
        return [(q, r) for q, r in targets if q and r]
