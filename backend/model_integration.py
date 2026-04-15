"""
Model Integration — SAM 3 / SAM 3.1 via facebookresearch/sam3 package.

Loads SAM 3 and SAM 3.1 checkpoints from HuggingFace (gated), builds the image
model with interactive-instance support enabled, and runs point-prompt,
text-prompt, and box-prompt segmentation through the Sam3Processor API.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from PIL import Image


class ModelManager:
    """Manage SAM 3 / SAM 3.1 models for interactive and batch annotation."""

    # Full pretrained catalog — single source of truth used by list_models()
    _PRETRAINED_CATALOG = [
        {"id": "sam3",  "name": "SAM 3",   "type": "sam3", "pretrained": True},
        {"id": "sam31", "name": "SAM 3.1", "type": "sam3", "pretrained": True},
    ]

    # HuggingFace gated repos. Both require a user access token.
    _HF_GATED_MODELS = {
        "sam3":  ("facebook/sam3",   "sam3.pt"),
        "sam31": ("facebook/sam3.1", "sam3.1_multiplex.pt"),
    }

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        # Records the outcome of the most recent load_pretrained() call so the
        # download endpoint can surface errors even when the failed entry is
        # not kept in loaded_models.
        self.last_load_result: Dict[str, Dict[str, Any]] = {}
        # Set by _run_inference when text grounding errored, so the single-image
        # endpoint can return a real 422 instead of a silent empty list.
        self._last_text_error: Optional[str] = None

    # ── Discovery ────────────────────────────────────────────────────────────

    def list_models(self) -> List[Dict[str, Any]]:
        """List SAM 3 / SAM 3.1 entries — loaded, downloaded, or catalog."""
        models: List[Dict[str, Any]] = []
        seen_ids: set = set()

        # 1. In-memory loaded models
        for model_id, info in self.loaded_models.items():
            seen_ids.add(model_id)
            is_loaded = info.get("model") is not None
            path_str = info.get("path")
            path_on_disk = bool(path_str) and Path(path_str).exists()
            models.append({
                "id": model_id,
                "name": info.get("name") or model_id,
                "type": info.get("type", "sam3"),
                "loaded": is_loaded,
                "pretrained": info.get("pretrained", True),
                "classes": info.get("classes", []),
                "error": info.get("error"),
                "downloaded": is_loaded or path_on_disk,
            })

        # 2. Checkpoints present on disk in workspace/models/
        for entry in self._PRETRAINED_CATALOG:
            if entry["id"] in seen_ids:
                continue
            _, filename = self._HF_GATED_MODELS[entry["id"]]
            local_path = self.models_dir / filename
            if local_path.exists():
                seen_ids.add(entry["id"])
                models.append({
                    **entry,
                    "loaded": False,
                    "downloaded": True,
                    "path": str(local_path),
                })

        # 3. Catalog entries not yet downloaded
        for entry in self._PRETRAINED_CATALOG:
            if entry["id"] not in seen_ids:
                models.append({**entry, "loaded": False, "downloaded": False})

        return models

    # ── Loading ──────────────────────────────────────────────────────────────

    def load_model(self, model_path: str, model_type: str) -> str:
        """Load a custom SAM 3 / SAM 3.1 checkpoint file from disk."""
        model_path = Path(model_path)
        model_id = model_path.stem

        model_info: Dict[str, Any] = {
            "id": model_id,
            "name": model_path.name,
            "type": "sam3",
            "path": str(model_path),
            "model": None,
            "processor": None,
            "classes": [],
        }

        try:
            self._build_sam3_model(str(model_path), model_info)
        except Exception as exc:
            model_info["error"] = str(exc)

        self.loaded_models[model_id] = model_info
        return model_id

    def load_pretrained(
        self, model_type: str, model_name: str, hf_token: Optional[str] = None
    ) -> str:
        """Download a gated SAM 3 / 3.1 checkpoint from HuggingFace and build the model."""
        model_id = model_name
        model_info: Dict[str, Any] = {
            "id": model_id,
            "name": model_name,
            "type": "sam3",
            "pretrained": True,
            "loaded": True,
            "model": None,
            "processor": None,
            "classes": [],
        }

        try:
            self._load_pretrained_sam3(model_name, model_info, hf_token=hf_token)
        except Exception as exc:
            model_info["error"] = str(exc)

        # Only keep entries that produced a usable model or at least a file on disk.
        has_model = model_info.get("model") is not None
        has_path = (
            bool(model_info.get("path"))
            and Path(model_info["path"]).exists()
        )
        if has_model or has_path:
            self.loaded_models[model_id] = model_info
        else:
            self.loaded_models.pop(model_id, None)

        self.last_load_result[model_id] = {
            "error": model_info.get("error"),
            "has_model": has_model,
            "has_path": has_path,
        }
        return model_id

    def unload(self, model_id: str) -> bool:
        """Drop a model from the in-memory registry."""
        return self.loaded_models.pop(model_id, None) is not None

    def _load_pretrained_sam3(
        self, model_name: str, model_info: Dict, hf_token: Optional[str] = None
    ) -> Dict:
        """Download a SAM 3 / 3.1 checkpoint via huggingface_hub, then build the model."""
        entry = self._HF_GATED_MODELS.get(model_name)
        if entry is None:
            model_info["error"] = f"Unknown SAM 3 model: {model_name}"
            model_info["loaded"] = False
            return model_info

        repo_id, filename = entry
        local_model_path = self.models_dir / filename

        if not local_model_path.exists():
            if not hf_token:
                model_info["error"] = (
                    f"{model_name} is a gated HuggingFace model. "
                    "Paste your HuggingFace access token (hf.co/settings/tokens) "
                    "in the Models page to download it."
                )
                model_info["loaded"] = False
                return model_info
            try:
                from huggingface_hub import hf_hub_download
            except ImportError:
                model_info["error"] = (
                    "huggingface_hub is not installed — add it to requirements.txt"
                )
                model_info["loaded"] = False
                return model_info
            try:
                dl_path = Path(hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    token=hf_token,
                    local_dir=str(self.models_dir),
                ))
                if dl_path != local_model_path:
                    import shutil
                    shutil.copy2(dl_path, local_model_path)
            except Exception as dl_err:
                if local_model_path.exists():
                    local_model_path.unlink()
                model_info["error"] = f"HuggingFace download failed: {dl_err}"
                model_info["loaded"] = False
                return model_info

        model_info["path"] = str(local_model_path)

        try:
            self._build_sam3_model(str(local_model_path), model_info)
        except Exception as exc:
            model_info["error"] = f"Failed to build SAM 3 model: {exc}"
            model_info["loaded"] = False
            return model_info

        model_info["loaded"] = True
        return model_info

    def _build_sam3_model(self, checkpoint_path: str, model_info: Dict) -> None:
        """Build a SAM 3 image model + processor from a local checkpoint.

        Instance interactivity is always enabled so the SAM wand can call
        `model.predict_inst()` with point prompts.
        """
        try:
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            import sam3 as _sam3_pkg
        except ImportError as exc:
            raise RuntimeError(
                "The `sam3` package is not installed. Install it with "
                "`pip install git+https://github.com/facebookresearch/sam3.git`"
            ) from exc

        bpe_path = (
            Path(_sam3_pkg.__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        )
        if not bpe_path.exists():
            raise RuntimeError(
                f"BPE vocab not found at {bpe_path} — the sam3 package install is incomplete."
            )

        device = self._get_device()

        model = build_sam3_image_model(
            bpe_path=str(bpe_path),
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
            enable_inst_interactivity=True,
            device=device,
            eval_mode=True,
        )
        processor = Sam3Processor(model)

        model_info["model"] = model
        model_info["processor"] = processor

    # ── Auto-load ────────────────────────────────────────────────────────────

    def _try_autoload(self, model_id: str) -> None:
        """Load a SAM 3 / 3.1 model into memory on demand, given its catalog ID."""
        if model_id in self.loaded_models and self.loaded_models[model_id].get("model"):
            return

        entry = self._HF_GATED_MODELS.get(model_id)
        if not entry:
            return

        _, filename = entry
        local_path = self.models_dir / filename
        if not local_path.exists():
            return

        catalog = next((e for e in self._PRETRAINED_CATALOG if e["id"] == model_id), None)
        info: Dict[str, Any] = {
            "id": model_id,
            "name": catalog["name"] if catalog else filename,
            "type": "sam3",
            "pretrained": True,
            "path": str(local_path),
            "model": None,
            "processor": None,
            "classes": [],
        }
        try:
            self._build_sam3_model(str(local_path), info)
            info["loaded"] = True
        except Exception as exc:
            info["error"] = str(exc)
            info["loaded"] = False
        self.loaded_models[model_id] = info

    # ── Device ───────────────────────────────────────────────────────────────

    def _get_device(self) -> str:
        """Return the best available torch device as a string."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    # ── Single-image inference ───────────────────────────────────────────────

    def annotate_single_image(
        self,
        model_id: str,
        dataset_path: Path,
        format_name: str,
        image_id: str,
        confidence_threshold: float = 0.5,
        prompt_point: Optional[tuple] = None,
        prompt_points: Optional[List[Dict[str, Any]]] = None,
        text_prompt: Optional[str] = None,
        image_path_hint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run SAM 3 on a single image with point, text, or auto prompts."""
        self._last_text_error = None

        if model_id not in self.loaded_models or not self.loaded_models[model_id].get("model"):
            self._try_autoload(model_id)

        if model_id not in self.loaded_models:
            return []

        model_info = self.loaded_models[model_id]
        model = model_info.get("model")
        model_type = model_info.get("type", "sam3")
        if not model:
            return []

        dataset_path = Path(dataset_path)

        IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
        image_file: Optional[Path] = None

        if image_path_hint:
            candidate = dataset_path / image_path_hint
            if candidate.exists():
                image_file = candidate

        if not image_file:
            for ext in IMAGE_EXTENSIONS:
                for found in dataset_path.glob(f"**/{image_id}{ext}"):
                    image_file = found
                    break
                if image_file:
                    break

        if not image_file:
            return []

        # Legacy single-point callers → new prompt_points shape
        if prompt_points is None and prompt_point is not None:
            prompt_points = [{"x": prompt_point[0], "y": prompt_point[1], "label": 1}]

        try:
            return self._run_inference(
                model, model_type, image_file, confidence_threshold,
                prompt_points=prompt_points,
                text_prompt=text_prompt,
                model_classes=model_info.get("classes") or [],
            )
        except Exception:
            return []

    # ── Batch auto-annotation ────────────────────────────────────────────────

    def auto_annotate(
        self,
        model_id: str,
        dataset_path: Path,
        format_name: str,
        confidence_threshold: float = 0.5,
        text_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Auto-annotate all images in a dataset using a text prompt (batch)."""
        if model_id not in self.loaded_models:
            self._try_autoload(model_id)
        if model_id not in self.loaded_models:
            return {"error": "Model not loaded"}

        model_info = self.loaded_models[model_id]
        model = model_info.get("model")
        model_type = model_info.get("type", "sam3")

        if not model:
            return {"error": "Model not available"}

        dataset_path = Path(dataset_path)
        results: Dict[str, Any] = {
            "annotated_count": 0,
            "failed_count": 0,
            "images": [],
        }

        IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
        image_files: List[Path] = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(dataset_path.glob(f"**/*{ext}"))
            image_files.extend(dataset_path.glob(f"**/*{ext.upper()}"))

        classes = self._get_dataset_classes(dataset_path, format_name)

        for image_file in image_files:
            try:
                annotations = self._run_inference(
                    model, model_type, image_file, confidence_threshold,
                    text_prompt=text_prompt,
                    model_classes=model_info.get("classes") or [],
                )

                if annotations:
                    self._save_annotations(
                        dataset_path, format_name, image_file, annotations, classes
                    )
                    results["annotated_count"] += 1
                    results["images"].append({
                        "filename": image_file.name,
                        "annotations": len(annotations),
                    })
            except Exception:
                results["failed_count"] += 1

        return results

    def annotate_with_new_classes(
        self,
        model_id: str,
        dataset_path: Path,
        format_name: str,
        new_classes: List[str],
    ) -> Dict[str, Any]:
        """Annotate a dataset with a list of text prompts (one per class)."""
        if model_id not in self.loaded_models:
            self._try_autoload(model_id)
        if model_id not in self.loaded_models:
            return {"error": "Model not loaded"}

        if not new_classes:
            return {"error": "No classes provided"}

        # SAM 3 text grounding accepts one prompt at a time via Sam3Processor;
        # run the dataset once per class and merge results.
        total_annotated = 0
        total_failed = 0
        for cls_name in new_classes:
            results = self.auto_annotate(
                model_id, dataset_path, format_name,
                confidence_threshold=0.5,
                text_prompt=cls_name,
            )
            total_annotated += results.get("annotated_count", 0)
            total_failed += results.get("failed_count", 0)

        return {
            "annotated_count": total_annotated,
            "failed_count": total_failed,
            "added_classes": new_classes,
        }

    # ── SAM 3 inference core ─────────────────────────────────────────────────

    def _run_inference(
        self,
        model,
        model_type: str,
        image_path: Path,
        confidence_threshold: float,
        prompt_points: Optional[List[Dict[str, Any]]] = None,
        text_prompt: Optional[str] = None,
        model_classes: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Run SAM 3 inference on one image. Returns polygon annotations."""
        annotations: List[Dict[str, Any]] = []

        try:
            import numpy as np
            import cv2
        except ImportError:
            return annotations

        # Locate the processor stashed alongside the model in loaded_models.
        processor = None
        for info in self.loaded_models.values():
            if info.get("model") is model:
                processor = info.get("processor")
                break
        if processor is None:
            return annotations

        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        h, w = image_array.shape[:2]

        def _binary_to_polygons(binary: "np.ndarray", conf: float,
                                cls_name: str = "object", cls_id: int = 0) -> List[Dict]:
            out: List[Dict] = []
            contours, _ = cv2.findContours(
                binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                if contour.shape[0] < 3:
                    continue
                pts = contour.squeeze().astype(float)
                if pts.ndim != 2:
                    continue
                out.append({
                    "type": "polygon",
                    "class_id": int(cls_id),
                    "class_name": cls_name,
                    "confidence": float(conf),
                    "points": pts.flatten().tolist(),
                })
            return out

        try:
            state = processor.set_image(image)
        except Exception as exc:
            print(f"[sam3] set_image failed: {exc}")
            return annotations

        # ── Text prompt (concept segmentation) ───────────────────────────
        if text_prompt is not None:
            texts = [t.strip() for t in text_prompt.split(",") if t.strip()] or [text_prompt]
            for cls_id, text in enumerate(texts):
                try:
                    out = processor.set_text_prompt(prompt=text, state=state)
                except Exception as exc:
                    self._last_text_error = f"{type(exc).__name__}: {exc}"
                    print(f"[sam3_text] set_text_prompt failed: {exc}")
                    continue

                masks = out.get("masks")
                scores = out.get("scores")
                if masks is None:
                    continue

                # masks may be np.ndarray (N,H,W) or torch.Tensor
                masks_np = _to_numpy(masks)
                scores_np = _to_numpy(scores) if scores is not None else None

                if masks_np is None:
                    continue

                # Normalize to (N, H, W)
                if masks_np.ndim == 2:
                    masks_np = masks_np[None, ...]

                for i, mask in enumerate(masks_np):
                    score = float(scores_np[i]) if scores_np is not None and i < len(scores_np) else 1.0
                    if score < confidence_threshold:
                        continue
                    binary = (mask > 0.5).astype("uint8")
                    if binary.sum() == 0:
                        continue
                    annotations.extend(_binary_to_polygons(binary, score, text, cls_id))

            if not annotations and self._last_text_error is None:
                # No matches — leave _last_text_error unset so callers don't 422.
                pass
            return annotations

        # ── Point prompt (SAM-1-style interactive) ───────────────────────
        if prompt_points:
            try:
                import numpy as np  # noqa: F811
            except ImportError:
                return annotations

            pts_px = np.array(
                [[int(p["x"] * w), int(p["y"] * h)] for p in prompt_points],
                dtype=np.float32,
            )
            labels = np.array(
                [int(p.get("label", 1)) for p in prompt_points],
                dtype=np.int64,
            )

            try:
                masks, scores, _logits = model.predict_inst(
                    state,
                    point_coords=pts_px,
                    point_labels=labels,
                    multimask_output=True,
                )
            except Exception as exc:
                print(f"[sam3_point] predict_inst failed: {exc}")
                return annotations

            masks_np = _to_numpy(masks)
            scores_np = _to_numpy(scores)
            if masks_np is None or masks_np.size == 0:
                return annotations

            if masks_np.ndim == 2:
                masks_np = masks_np[None, ...]

            # Pick the single best-scored mask
            if scores_np is not None and scores_np.size > 0:
                best = int(scores_np.argmax())
                best_score = float(scores_np.flatten()[best])
            else:
                best = 0
                best_score = 1.0

            if best >= len(masks_np):
                best = 0

            binary = (masks_np[best] > 0.5).astype("uint8")
            if binary.sum() > 0:
                annotations.extend(_binary_to_polygons(binary, best_score))
            return annotations

        # ── No prompt: not supported for SAM 3 image model ──────────────
        return annotations

    # ── Format-specific annotation I/O ───────────────────────────────────────

    def _get_dataset_classes(self, dataset_path: Path, format_name: str) -> List[str]:
        """Read class names from a dataset's metadata file."""
        classes: List[str] = []

        if format_name in ["yolo", "yolov5", "yolov8", "yolov9", "yolov10", "yolov11", "yolov12"]:
            for yaml_file in list(dataset_path.glob("*.yaml")) + list(dataset_path.glob("*.yml")):
                try:
                    with open(yaml_file) as f:
                        config = yaml.safe_load(f)
                        if "names" in config:
                            if isinstance(config["names"], dict):
                                classes = list(config["names"].values())
                            else:
                                classes = config["names"]
                        break
                except Exception:
                    pass

        elif format_name == "coco":
            for json_file in list(dataset_path.glob("*.json")) + list(dataset_path.glob("annotations/*.json")):
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                        if "categories" in data:
                            classes = [cat["name"] for cat in data["categories"]]
                        break
                except Exception:
                    pass

        return classes

    def _save_annotations(
        self,
        dataset_path: Path,
        format_name: str,
        image_path: Path,
        annotations: List[Dict],
        classes: List[str],
    ) -> None:
        """Write annotations to the dataset in the correct on-disk format."""
        with Image.open(image_path) as img:
            width, height = img.size

        image_id = image_path.stem

        if format_name in ["yolo", "yolov5", "yolov8", "yolov9", "yolov10", "yolov11", "yolov12"]:
            try:
                rel_parts = image_path.relative_to(dataset_path).parts
                img_idx = next(
                    (i for i, p in enumerate(rel_parts) if p.lower() in ("images", "imgs")),
                    None,
                )
                if img_idx is not None:
                    labels_dir = dataset_path.joinpath(*rel_parts[:img_idx]) / "labels"
                else:
                    raise ValueError("no 'images' segment in path")
            except Exception:
                labels_dir = dataset_path / "labels"
                if (dataset_path / "train" / "labels").exists():
                    labels_dir = dataset_path / "train" / "labels"
            labels_dir.mkdir(parents=True, exist_ok=True)

            class_to_id = {name: idx for idx, name in enumerate(classes)}
            new_classes = []
            for ann in annotations:
                if ann["class_name"] not in class_to_id:
                    class_to_id[ann["class_name"]] = len(classes) + len(new_classes)
                    new_classes.append(ann["class_name"])

            if new_classes:
                self._update_yolo_classes(dataset_path, classes + new_classes)

            label_file = labels_dir / f"{image_id}.txt"
            with open(label_file, "w") as f:
                for ann in annotations:
                    class_id = class_to_id.get(ann["class_name"], 0)

                    if ann.get("type") == "polygon":
                        raw_pts = ann.get("points") or (ann.get("segmentation", [[]])[0] if ann.get("segmentation") else [])
                        normalized = []
                        for i in range(0, len(raw_pts) - 1, 2):
                            normalized.append(raw_pts[i] / width)
                            normalized.append(raw_pts[i + 1] / height)
                        if normalized:
                            f.write(f"{class_id} " + " ".join(f"{p:.6f}" for p in normalized) + "\n")
                    elif ann.get("bbox"):
                        bbox = ann["bbox"]
                        x_center = (bbox[0] + bbox[2] / 2) / width
                        y_center = (bbox[1] + bbox[3] / 2) / height
                        bw = bbox[2] / width
                        bh = bbox[3] / height
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

    def _update_yolo_classes(self, dataset_path: Path, classes: List[str]) -> None:
        """Keep the dataset's data.yaml in sync with new class names."""
        yaml_file: Optional[Path] = None
        for yf in list(dataset_path.glob("*.yaml")) + list(dataset_path.glob("*.yml")):
            yaml_file = yf
            break

        if not yaml_file:
            yaml_file = dataset_path / "data.yaml"

        if yaml_file.exists():
            with open(yaml_file) as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {
                "path": str(dataset_path.absolute()),
                "train": "images",
                "val": "images",
            }

        config["names"] = {i: name for i, name in enumerate(classes)}
        config["nc"] = len(classes)

        with open(yaml_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False)


def _to_numpy(value: Any) -> Optional[Any]:
    """Best-effort conversion of a SAM 3 output (tensor or ndarray) to numpy."""
    if value is None:
        return None
    try:
        import numpy as np
    except ImportError:
        return None
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        try:
            return value.detach().cpu().numpy()
        except Exception:
            pass
    if hasattr(value, "cpu"):
        try:
            return value.cpu().numpy()
        except Exception:
            pass
    try:
        return np.asarray(value)
    except Exception:
        return None
