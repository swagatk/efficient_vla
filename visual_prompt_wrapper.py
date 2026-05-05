import cv2
import numpy as np
import torch
import os
from PIL import Image

# --- TEXT HINT CONFIGURATION ---
# Controls the formatting of the text hints injected into instructions.
# Options:
# "highlight" - e.g., "[object, highlighted by the magenta box]"
# "spatial"   - e.g., "[object, located at the top right]"
# "both"      - e.g., "[object, highlighted by the magenta box and located at the top right]"
# "coords"    - e.g., "[object, box: 10 136 44 186]"
TEXT_HINT_MODE = "both"  # Change this to switch text hint styles
# -------------------------------

# --- GROUNDING MODEL CONFIGURATION ---
# Controls which model(s) are used for object grounding.
# Options:
# "dino"    - Use only GroundingDINO
# "fastsam" - Use only FastSAM
# "both"    - Use both and fuse them
GROUNDING_MODE = "dino"  # Change this to switch grounding models
# -------------------------------

try:
    from groundingdino.util.inference import Model
except ImportError:
    print("Warning: GroundingDINO not found. Please install groundingdino-py.")

try:
    from fastsam import FastSAM
except ImportError as e:
    try:
        from ultralytics import FastSAM
    except ImportError:
        print(f"Warning: FastSAM import failed: {e}. FastSAM requires the 'ultralytics' package. Please run: pip install ultralytics")
        FastSAM = None

class VisualPromptingWrapper:
    def __init__(self, use_image_box=True, use_text_hint=True, device="cuda", box_style="edge"):
        self.use_image_box = use_image_box
        self.use_text_hint = use_text_hint
        self.device = device
        self.box_style = box_style
        self._last_dino_boxes = []
        self._last_fastsam_boxes = []
        self._last_fused_boxes = []
        self._last_prompts_to_apply = {}
        
        # Point to the exact path in your GIT/efficient_vla folder where weights are downloaded
        base_dir = os.path.expanduser("~/GIT/efficient_vla")
        weights_dir = os.path.expanduser("~/model_weights")
        
        # 1. Initialize Grounding Model (GroundingDINO)
        self.model = None
        if GROUNDING_MODE in ["dino", "both"]:
            config_path = os.path.join(base_dir, "GroundingDINO_SwinT_OGC.py")
            weights_path = os.path.join(weights_dir, "groundingdino_swint_ogc.pth")
            
            if os.path.exists(config_path) and os.path.exists(weights_path):
                try:
                    self.model = Model(model_config_path=config_path, model_checkpoint_path=weights_path, device=self.device)
                    print("GroundingDINO loaded successfully.")
                except Exception as e:
                    print(f"Error loading GroundingDINO: {e}")
            else:
                print(f"Warning: {config_path} or {weights_path} not found. Ensure weights are downloaded.")

        # 2. Initialize FastSAM
        self.fastsam_model = None
        if GROUNDING_MODE in ["fastsam", "both"]:
            if FastSAM:
                fastsam_weights_path = os.path.join(weights_dir, "FastSAM-x.pt")
    
                # Check for different weight namings
                possible_weights = ["FastSAM-x.pt", "fastsam-x.pt", "FastSAM-s.pt", "fastsam-s.pt"]
                for pw in possible_weights:
                    p = os.path.join(weights_dir, pw)
                    if os.path.exists(p):
                        fastsam_weights_path = p
                        break
                
                # Automatically download weights if they are missing
                if not os.path.exists(fastsam_weights_path):
                    print(f"[FastSAM Init] Downloading weights to {fastsam_weights_path}...")
                    try:
                        os.makedirs(weights_dir, exist_ok=True)
                        import urllib.request
                        urllib.request.urlretrieve("https://github.com/ultralytics/assets/releases/download/v0.0.0/FastSAM-x.pt", fastsam_weights_path)
                        print("[FastSAM Init] Download successful.")
                    except Exception as e:
                        print(f"[FastSAM Init] Failed to download FastSAM weights: {e}")
    
                if os.path.exists(fastsam_weights_path):
                    try:
                        print(f"[FastSAM Init] Loading model from {fastsam_weights_path}...")
                        self.fastsam_model = FastSAM(fastsam_weights_path)
                        print("[FastSAM Init] Model loaded successfully.")
                    except Exception as e:
                        print(f"[FastSAM Init] Error loading FastSAM: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"[FastSAM Init] Warning: {fastsam_weights_path} not found. FastSAM will not be used.")
            else:
                print("[FastSAM Init] FastSAM library is not imported. Skipping FastSAM initialization.")
            
        self._fastsam_warned = False

    def get_grounding_box_for_target(self, image_rgb: np.ndarray, target: str, return_all: bool = False):
        """
        Query GroundingDINO for a bounding box of a specific target object.
        Can return all detected boxes or only the one with the highest confidence.
        """
        if self.model is None:
            return [] if return_all else None

        query = target.lower() + "."

        try:
            outputs = self.model.predict_with_caption(
                image=image_rgb,
                caption=query,
                box_threshold=0.4,
                text_threshold=0.3
            )
            
            if len(outputs) == 2:
                detections, phrases = outputs
                if len(detections) == 0:
                    return [] if return_all else None
                
                xyxy = detections.xyxy
                conf = detections.confidence
                if hasattr(xyxy, 'cpu'):
                    xyxy = xyxy.cpu().numpy()
                    conf = conf.cpu().numpy() if hasattr(conf, 'cpu') else conf
                
                # Sort by confidence descending
                sorted_indices = np.argsort(-conf)
                all_boxes = xyxy[sorted_indices].astype(int).tolist()

                if not return_all:
                    return all_boxes[0]
                return all_boxes
            else:
                # Handle older groundingdino API format
                boxes, logits, phrases = outputs
                if len(boxes) == 0:
                    return [] if return_all else None

                h, w = image_rgb.shape[:2]
                all_boxes = []
                
                conf = logits.cpu().numpy() if hasattr(logits, 'cpu') else logits
                sorted_indices = np.argsort(-conf)
                
                for idx in sorted_indices:
                    box = boxes[idx]
                    cx, cy, bw, bh = box.cpu().numpy() if hasattr(box, 'cpu') else box
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    all_boxes.append([x1, y1, x2, y2])

                if not return_all:
                    return all_boxes[0]
                return all_boxes
            
        except Exception as e:
            print(f"GroundingDINO prediction error: {e}")
            return [] if return_all else None

    def get_all_fastsam_boxes(self, image_rgb: np.ndarray, target: str = None):
        """
        Query FastSAM for all detected object bounding boxes ("everything" mode or text prompted).
        """
        if self.fastsam_model is None:
            return []
        try:
            # Convert to PIL Image so YOLO strictly treats the input as RGB (it assumes numpy arrays are BGR)
            image_pil = Image.fromarray(image_rgb)
            
            kwargs = {
                "device": self.device,
                "retina_masks": True,
                "imgsz": 1024,
                "conf": 0.9,
                "iou": 0.9,
                "verbose": False
            }
            if target:
                kwargs["texts"] = target.lower()

            everything_results = self.fastsam_model(image_pil, **kwargs)
            if not everything_results:
                print("[FastSAM] No objects detected (empty results).")
                return []
            
            # The result from FastSAM is a list of ultralytics.yolo.engine.results.Results
            # We take the first one for our single image.
            result = everything_results[0]
            if result.boxes is None or len(result.boxes) == 0:
                target_str = f" for '{target}'" if target else ""
                print(f"[FastSAM] No objects detected{target_str} (no boxes).")
                return []

            boxes_tensor = result.boxes.xyxy
            conf_tensor = result.boxes.conf
            
            boxes_np = boxes_tensor.cpu().numpy()
            conf_np = conf_tensor.cpu().numpy()
            
            # Sort FastSAM boxes by confidence descending
            sorted_indices = np.argsort(-conf_np)
            boxes_list = boxes_np[sorted_indices].astype(int).tolist()

            target_str = f" for '{target}'" if target else ""
            print(f"[FastSAM] Detected {len(boxes_list)} total objects{target_str}.")
            return boxes_list

        except Exception as e:
            print(f"FastSAM prediction error: {e}")
            return []

    def calculate_overlap_metrics(self, box1, box2):
        """Calculates Intersection over Union (IoU) and Intersection over Area of Box 2 (IoA)."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        
        iou = inter_area / union_area if union_area > 0 else 0
        
        # Calculate how much of box2 is contained within box1 (Intersection over Area of box 2)
        ioa2 = inter_area / box2_area if box2_area > 0 else 0
        return iou, ioa2

    def fuse_boxes(self, dino_boxes, fastsam_boxes, expected_count, global_used_boxes=None, iou_threshold=0.3):
        """
        Fuses detections from DINO and FastSAM using a constrained union-merge strategy,
        while preventing reassignment of boxes already used by other targets.
        """
        if global_used_boxes is None:
            global_used_boxes = []

        if not dino_boxes and not fastsam_boxes:
            return []

        final_boxes = []
        used_fastsam_indices = set()

        # Filter out boxes that are already assigned to previous targets
        def is_used(box):
            for g_box in global_used_boxes:
                iou, ioa = self.calculate_overlap_metrics(box, g_box)
                if iou > 0.5 or ioa > 0.8:
                    return True
            return False

        valid_dino_boxes = [b for b in dino_boxes if not is_used(b)]
        valid_fastsam_boxes = [b for b in fastsam_boxes if not is_used(b)]

        # Step 1: Match valid DINO boxes with best-overlapping valid FastSAM boxes
        if valid_dino_boxes:
            for d_box in valid_dino_boxes:
                best_iou = -1
                best_fs_idx = -1
                for i, fs_box in enumerate(valid_fastsam_boxes):
                    if i in used_fastsam_indices:
                        continue
                    iou, ioa_fastsam = self.calculate_overlap_metrics(d_box, fs_box)
                    
                    # If standard IoU matches OR FastSAM is mostly contained inside a sloppy DINO box
                    if iou > best_iou or ioa_fastsam > 0.8:
                        best_iou = iou
                        best_fs_idx = i

                if best_fs_idx != -1 and (best_iou > iou_threshold or ioa_fastsam > 0.8):
                    # Match found, prefer the FastSAM box for its tighter boundary
                    final_boxes.append(valid_fastsam_boxes[best_fs_idx])
                    used_fastsam_indices.add(best_fs_idx)
                else:
                    # No good overlap, keep the DINO box
                    final_boxes.append(d_box)
        
        # Step 2: If we still need more boxes, add remaining high-confidence FastSAM boxes
        if len(final_boxes) < expected_count:
            for i, fs_box in enumerate(valid_fastsam_boxes):
                if i not in used_fastsam_indices:
                    final_boxes.append(fs_box)
                    if len(final_boxes) >= expected_count:
                        break
        
        # Step 3: Ensure we don't exceed the expected count
        return final_boxes[:expected_count]

    def _get_spatial_hint(self, box, img_w, img_h):
        """Converts a bounding box into a descriptive spatial string (e.g., 'top right')."""
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        
        if cx < img_w / 3:
            h_pos = "left"
        elif cx > 2 * img_w / 3:
            h_pos = "right"
        else:
            h_pos = "center"
            
        if cy < img_h / 3:
            v_pos = "top"
        elif cy > 2 * img_h / 3:
            v_pos = "bottom"
        else:
            v_pos = "middle"
            
        if v_pos == "middle" and h_pos == "center":
            return "in the center"
        if v_pos == "middle":
            return f"on the {h_pos}"
        if h_pos == "center":
            return f"at the {v_pos}"
        return f"at the {v_pos} {h_pos}"

    def apply_prompts(self, observation: dict, update_grounding: bool = True):
        """
        Intercepts the observation, runs DINO and FastSAM, fuses the detections,
        and returns the observation with prompted image and instruction.
        """
        img = observation['image'] # Expected (H, W, 3) numpy array
        instruction = observation['instruction']
        
        targets = self.extract_target_objects(instruction)
        if not targets:
            return observation

        from collections import defaultdict
        grouped_targets = defaultdict(list)
        for query, replace in targets:
            grouped_targets[replace].append(query)

        prompted_img = img.copy()
        debug_img = img.copy()
        new_instruction = instruction
        all_fused_boxes = []
        all_dino_boxes = []
        all_fastsam_boxes = []
        global_used_boxes = []
        prompts_to_apply = {} # Key: replace_target, Value: list of boxes

        if update_grounding:
            for replace_target, queries in grouped_targets.items():
                query_target = queries[0]
                expected_count = len(queries)
    
                dino_boxes = []
                if GROUNDING_MODE in ["dino", "both"]:
                    dino_boxes = self.get_grounding_box_for_target(img, query_target, return_all=True)
                    if dino_boxes:
                        all_dino_boxes.extend(dino_boxes)
    
                fastsam_boxes = []
                if GROUNDING_MODE in ["fastsam", "both"]:
                    fastsam_boxes = self.get_all_fastsam_boxes(img, target=query_target)
                    if fastsam_boxes:
                        all_fastsam_boxes.extend(fastsam_boxes)
    
                fused_boxes = self.fuse_boxes(
                    dino_boxes, fastsam_boxes, expected_count, global_used_boxes=global_used_boxes
                )
                
                if fused_boxes:
                    prompts_to_apply[replace_target] = fused_boxes
                    all_fused_boxes.extend(fused_boxes)
                    global_used_boxes.extend(fused_boxes) # Update global tracker
            
            self._last_prompts_to_apply = prompts_to_apply
            self._last_fused_boxes = all_fused_boxes
            self._last_dino_boxes = all_dino_boxes
            self._last_fastsam_boxes = all_fastsam_boxes
        else:
            # Reuse cached boxes for this step to reduce latency
            prompts_to_apply = self._last_prompts_to_apply
            all_fused_boxes = self._last_fused_boxes
            all_dino_boxes = self._last_dino_boxes
            all_fastsam_boxes = self._last_fastsam_boxes

        if self.use_image_box:
            if self.box_style == "edge":
                for box in all_fused_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(prompted_img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            elif self.box_style == "filled":
                overlay = prompted_img.copy()
                for box in all_fused_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 255), -1)
                prompted_img = cv2.addWeighted(overlay, 0.4, prompted_img, 0.6, 0)
            elif self.box_style == "mask":
                if all_fused_boxes:
                    mask = np.zeros_like(prompted_img)
                    for box in all_fused_boxes:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
                    # Darken background to 30% intensity
                    darkened = (prompted_img * 0.3).astype(np.uint8)
                    np.copyto(prompted_img, darkened, where=(mask == 0))
            
            # Always output base debug visualizers
            if update_grounding:
                for box in all_fastsam_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                for box in all_dino_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                print(f"[Visualization] Applied {self.box_style} boxes: {len(all_fastsam_boxes)} FastSAM, {len(all_dino_boxes)} DINO, {len(all_fused_boxes)} Fused.")
        
        observation['image'] = prompted_img
        observation['debug_image'] = debug_img
            
        if self.use_text_hint:
            for replace_target, boxes in prompts_to_apply.items():
                if not boxes:
                    continue
                
                hint_keyword = "box" if len(boxes) == 1 else "boxes"
                
                messages = []
                
                if TEXT_HINT_MODE in ["highlight", "both"] and self.use_image_box:
                    messages.append(f"highlighted by the magenta {hint_keyword}")
                    
                if TEXT_HINT_MODE in ["spatial", "both"] or (TEXT_HINT_MODE == "highlight" and not self.use_image_box):
                    h, w = img.shape[:2]
                    spatial_strs = [self._get_spatial_hint(b, w, h) for b in boxes]
                    messages.append(f"located {' and '.join(spatial_strs)}")
                    
                if TEXT_HINT_MODE == "coords":
                    box_strs = [f"{b[0]} {b[1]} {b[2]} {b[3]}" for b in boxes]
                    messages.append(f"{hint_keyword}: {', '.join(box_strs)}")
                    
                if messages:
                    hint_str = f"[{replace_target}, {' and '.join(messages)}]"
                else:
                    hint_str = f"[{replace_target}]"
                    
                new_instruction = new_instruction.replace(replace_target, hint_str, 1)
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
                singular = objects_str[:-1]
                # Handle "put both moka pots" -> query "moka pot" twice
                targets.append((singular, objects_str))
                targets.append((singular, objects_str))
            else:
                # Fallback for non-plural, e.g. "put both equipment"
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
