"""
CV Dataset Manager - Main FastAPI Application
A comprehensive tool for managing computer vision datasets
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import shutil
import uuid
import random
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

from dataset_parsers import DatasetParser
from format_converter import FormatConverter
from annotation_tools import AnnotationManager
from model_integration import ModelManager
from training import TrainingManager
from dataset_merger import DatasetMerger
from video_utils import VideoFrameExtractor, DuplicateDetector, CLIPEmbeddingManager
from augmentation import DatasetAugmenter

# Configuration - must be defined before functions that use them
WORKSPACE_DIR = Path("./workspace")
DATASETS_DIR = WORKSPACE_DIR / "datasets"
MODELS_DIR = WORKSPACE_DIR / "models"
EXPORTS_DIR = WORKSPACE_DIR / "exports"
SNAPSHOTS_DIR = WORKSPACE_DIR / "snapshots"
TEMP_DIR = WORKSPACE_DIR / "temp"
JOBS_FILE = WORKSPACE_DIR / "batch_jobs.json"

# Create directories
for d in [WORKSPACE_DIR, DATASETS_DIR, MODELS_DIR, EXPORTS_DIR, SNAPSHOTS_DIR, TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Active datasets storage
active_datasets: Dict[str, Any] = {}

# Initialize components
dataset_parser = DatasetParser()
format_converter = FormatConverter()
annotation_manager = AnnotationManager()
model_manager = ModelManager(MODELS_DIR)
training_manager = TrainingManager()
dataset_merger = DatasetMerger()
video_extractor = VideoFrameExtractor()
duplicate_detector = DuplicateDetector()
clip_manager = CLIPEmbeddingManager()
augmenter = DatasetAugmenter()

METADATA_FILENAME = "dataset_metadata.json"


def _save_dataset_metadata(dataset_id: str, info: dict):
    """Persist dataset info alongside the dataset files so it survives restarts."""
    try:
        meta_path = DATASETS_DIR / dataset_id / METADATA_FILENAME
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(info, f, indent=2, default=str)
    except Exception:
        pass


def _restore_datasets():
    """
    On startup: scan workspace/datasets/ for any folder that has a
    dataset_metadata.json sidecar and re-register it in active_datasets.
    Folders without metadata are re-parsed from scratch.
    """
    if not DATASETS_DIR.exists():
        return
    restored = 0
    for entry in DATASETS_DIR.iterdir():
        if not entry.is_dir():
            continue
        dataset_id = entry.name
        meta_path = entry / METADATA_FILENAME
        try:
            if meta_path.exists():
                with open(meta_path) as f:
                    info = json.load(f)
                info["id"] = dataset_id
                active_datasets[dataset_id] = info
                restored += 1
            else:
                # No sidecar — try to parse and create one
                info = dataset_parser.parse_dataset(entry, name=entry.name)
                info["id"] = dataset_id
                active_datasets[dataset_id] = info
                _save_dataset_metadata(dataset_id, info)
                restored += 1
        except Exception:
            pass  # Skip corrupt/incomplete datasets silently
    if restored:
        print(f"[startup] Restored {restored} dataset(s) from workspace.")


def _ensure_packages(packages: list[tuple[str, str]]) -> None:
    """Auto-install missing Python packages. packages = [(import_name, pip_name), ...]"""
    import importlib, subprocess, sys
    missing = [pip for imp, pip in packages if importlib.util.find_spec(imp) is None]
    if missing:
        print(f"[startup] Auto-installing missing packages: {missing}")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing,
            check=False,
        )


# Module-level GPU install status — polled by /api/device-info and the frontend banner
_gpu_status: dict = {
    "state": "unknown",   # "ready" | "installing" | "failed" | "no_gpu" | "unknown"
    "message": "",
    "gpu_name": "",
}


def _ensure_cuda_torch() -> None:
    """
    If NVIDIA GPU hardware is present but the installed PyTorch lacks CUDA support,
    automatically install the CUDA-enabled wheel and restart the backend process.
    Runs in a background daemon thread so server startup is never blocked.
    After a successful install, touches main.py (triggers WatchFiles reload) and
    calls os._exit(0) to kill the worker — WatchFiles spawns a fresh worker that
    picks up the new CUDA-enabled torch.
    """
    import subprocess, sys, os, pathlib, time

    global _gpu_status

    # 1. Detect GPU hardware via nvidia-smi (doesn't need torch)
    try:
        r = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5
        )
        if r.returncode != 0 or "GPU" not in r.stdout:
            _gpu_status["state"] = "no_gpu"
            return
        gpu_line = r.stdout.strip().splitlines()[0]
        _gpu_status["gpu_name"] = gpu_line
    except Exception:
        _gpu_status["state"] = "no_gpu"
        return  # nvidia-smi not found → no NVIDIA GPU

    # 2. Check if current torch already has CUDA
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            msg = f"CUDA ready: {name} (torch {torch.__version__}, CUDA {torch.version.cuda})"
            print(f"[GPU] {msg}")
            _gpu_status.update({"state": "ready", "message": msg, "gpu_name": name})
            return  # Already good
        torch_ver = torch.__version__
        print(f"[GPU] GPU found ({gpu_line}) but torch {torch_ver} has no CUDA support.")
    except ImportError:
        torch_ver = "not installed"
        print(f"[GPU] GPU found ({gpu_line}) but torch is not installed yet.")

    # 3. Auto-install CUDA-enabled PyTorch
    # Detect the right CUDA wheel from the driver version
    cuda_ver = "cu124"
    try:
        nv = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        drv = float(nv.stdout.strip().split(".")[0]) if nv.returncode == 0 else 0
        cuda_ver = "cu124" if drv >= 545 else "cu121"
    except Exception:
        pass

    wheel_url = f"https://download.pytorch.org/whl/{cuda_ver}"
    _gpu_status.update({
        "state": "installing",
        "message": f"Installing CUDA PyTorch ({cuda_ver})… backend will restart when done. "
                   f"Inference uses CPU until then.",
        "gpu_name": gpu_line,
    })
    print(f"[GPU] Auto-installing CUDA PyTorch ({cuda_ver}) from {wheel_url} ...")

    result = subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", wheel_url,
            "--upgrade", "--quiet",
        ],
        timeout=900,
        check=False,
    )

    if result.returncode == 0:
        print("[GPU] CUDA PyTorch installed. Triggering backend restart via WatchFiles...")
        _gpu_status["message"] = "Install done — restarting backend now…"
        try:
            # Touch this file → WatchFiles detects change → kills & respawns worker
            pathlib.Path(__file__).touch()
        except Exception:
            pass
        time.sleep(3)   # Give WatchFiles a moment to react
        os._exit(0)     # Kill worker process; WatchFiles spawns a fresh one with CUDA torch
    else:
        err = (result.stderr or result.stdout or "unknown error")[-400:]
        print(f"[GPU] CUDA PyTorch install failed: {err}")
        _gpu_status.update({
            "state": "failed",
            "message": f"Auto-install failed. Run manually:\n"
                       f"pip install torch torchvision --index-url {wheel_url}\n"
                       f"then restart the backend.",
        })


@asynccontextmanager
async def lifespan(app: FastAPI):
    import threading

    # Both checks run in background threads so startup never blocks.
    # _ensure_cuda_torch will call os.execv to restart the whole process once
    # the CUDA wheel finishes installing — that's fine from a daemon thread.
    threading.Thread(target=_ensure_cuda_torch, daemon=True).start()
    threading.Thread(
        target=_ensure_packages,
        args=([
            ("huggingface_hub", "huggingface_hub"),
            ("ultralytics",     "ultralytics"),
            ("cv2",             "opencv-python"),
            ("PIL",             "Pillow"),
            ("psutil",          "psutil"),
            # Pre-install CLIP so ultralytics doesn't auto-install it mid-inference
            # (which writes to venv and triggers a spurious WatchFiles reload)
            ("clip",            "git+https://github.com/ultralytics/CLIP.git"),
            ("ftfy",            "ftfy"),
        ],),
        daemon=True,
    ).start()
    _restore_datasets()
    _restore_jobs()
    yield


app = FastAPI(
    title="CV Dataset Manager",
    description="Professional Computer Vision Dataset Management Suite",
    version="3.0.0",
    lifespan=lifespan,
)


# Root route - shows API status
@app.get("/")
async def root():
    """Root endpoint showing API status and instructions"""
    return {
        "status": "running",
        "name": "CV Dataset Manager API",
        "version": "3.0.0",
        "message": "Backend is running! Open http://localhost:3000 for the UI.",
        "docs": "/docs",
        "health": "/api/health",
        "instructions": {
            "step1": "This is the backend API server",
            "step2": "The frontend UI runs on port 3000",
            "step3": "Run 'npm run dev' or 'pnpm dev' in the project root to start the frontend",
            "step4": "Then open http://localhost:3000 in your browser"
        }
    }

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class DatasetInfo(BaseModel):
    id: str
    name: str
    path: str
    format: str
    task_type: str
    num_images: int
    num_annotations: int
    classes: List[str]
    created_at: str
    splits: Optional[Dict[str, int]] = None


class AnnotationUpdate(BaseModel):
    image_id: str
    annotations: List[Dict[str, Any]]


class ConversionRequest(BaseModel):
    dataset_id: str
    target_format: str
    output_name: Optional[str] = None


class MergeRequest(BaseModel):
    dataset_ids: List[str]
    output_name: str
    output_format: str
    class_mapping: Optional[Dict[str, str]] = None


class TrainingConfig(BaseModel):
    dataset_id: str
    model_type: str  # "yolo" | "segmentation" | "classification" | "rfdetr"
    model_arch: str = "yolov8n"   # e.g. yolov8n, yolov8s, rfdetr_base, rfdetr_large
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    pretrained: bool = True
    device: str = "auto"
    # Advanced hyperparams (YOLO)
    lr0: float = 0.01
    lrf: float = 0.01
    optimizer: str = "SGD"
    patience: int = 50
    cos_lr: bool = False
    warmup_epochs: float = 3.0
    weight_decay: float = 0.0005
    mosaic: float = 1.0
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    flipud: float = 0.0
    fliplr: float = 0.5
    amp: bool = True
    dropout: float = 0.0




class SettingsConfig(BaseModel):
    models_path: Optional[str] = None
    datasets_path: Optional[str] = None
    output_path: Optional[str] = None
    use_gpu: bool = True
    gpu_device: str = "0"

class ClassAddRequest(BaseModel):
    dataset_id: str
    new_classes: List[str]
    use_model: bool = False
    model_id: Optional[str] = None


class ClassExtractRequest(BaseModel):
    dataset_id: str
    classes_to_extract: List[str]
    output_name: str


class ClassDeleteRequest(BaseModel):
    dataset_id: str
    classes_to_delete: List[str]


class ClassMergeRequest(BaseModel):
    dataset_id: str
    source_classes: List[str]
    target_class: str


class SplitRequest(BaseModel):
    dataset_id: str
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    output_name: Optional[str] = None
    shuffle: bool = True
    seed: Optional[int] = None


class AugmentationConfig(BaseModel):
    dataset_id: str
    output_name: str
    target_size: int  # Target number of images
    augmentations: Dict[str, Dict[str, Any]]  # e.g., {"flip_horizontal": {"enabled": true}, "rotate": {"angle": 15}}


class SortingAction(BaseModel):
    image_id: str
    action: str  # "keep" or "delete"


class AnnotationHistoryEntry(BaseModel):
    timestamp: str
    action: str
    annotation_data: Dict[str, Any]


# Dataset management state (active_datasets defined earlier)
sorting_sessions: Dict[str, Dict] = {}
annotation_history: Dict[str, List[Dict]] = {}  # dataset_id -> list of history entries
# In-memory image list cache: dataset_id -> list of image dicts
_images_cache: Dict[str, List] = {}

# Model download progress: model_id -> {"status": "downloading"|"done"|"error", "progress": 0-100, "error": str}
_download_status: Dict[str, Dict] = {}


# ============== DATASET LOADING ==============

@app.post("/api/datasets/load")
async def load_dataset(
    files: List[UploadFile] = File(...),
    dataset_name: str = None,
    format_hint: str = None
):
    """Load a dataset from uploaded files (supports zip, folders, or individual files)"""
    dataset_id = str(uuid.uuid4())
    dataset_path = DATASETS_DIR / dataset_id
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save uploaded files
        for file in files:
            file_path = dataset_path / file.filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            
            # Extract if zip
            if file.filename.endswith(".zip"):
                shutil.unpack_archive(file_path, dataset_path)
                os.remove(file_path)
        
        # Parse dataset
        dataset_info = dataset_parser.parse_dataset(
            dataset_path, 
            format_hint=format_hint,
            name=dataset_name or f"Dataset_{dataset_id[:8]}"
        )
        
        dataset_info["id"] = dataset_id
        active_datasets[dataset_id] = dataset_info
        _save_dataset_metadata(dataset_id, dataset_info)
        
        return {"success": True, "dataset": dataset_info}
    
    except Exception as e:
        shutil.rmtree(dataset_path, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/datasets")
async def list_datasets():
    """List all loaded datasets, re-parsing any that have stale zero counts."""
    for dataset_id, dataset in list(active_datasets.items()):
        if dataset.get("num_images", 0) == 0:
            try:
                dataset_path = DATASETS_DIR / dataset_id
                fresh = dataset_parser.parse_dataset(dataset_path, dataset["format"], dataset["name"])
                fresh["id"] = dataset_id
                active_datasets[dataset_id] = fresh
                _save_dataset_metadata(dataset_id, fresh)
            except Exception:
                pass
    return {"datasets": list(active_datasets.values())}


@app.get("/api/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Get detailed dataset information"""
    if dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = active_datasets[dataset_id]
    dataset_path = DATASETS_DIR / dataset_id
    
    # Get full dataset details including images
    details = dataset_parser.get_dataset_details(dataset_path, dataset["format"])
    return {"dataset": dataset, "details": details}


@app.get("/api/datasets/{dataset_id}/stats")
async def get_dataset_stats(dataset_id: str):
    """Get detailed statistics for a dataset including class distribution, splits, etc."""
    if dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = active_datasets[dataset_id]
    dataset_path = DATASETS_DIR / dataset_id

    # Calculate statistics (also re-parses to fix stale counts)
    stats = dataset_parser.get_dataset_details(dataset_path, dataset["format"])

    # If the stored metadata had stale zeros, refresh it now
    if dataset.get("num_images", 0) == 0 and stats.get("total_images", 0) > 0:
        fresh = dataset_parser.parse_dataset(dataset_path, dataset["format"], dataset["name"])
        fresh["id"] = dataset_id
        active_datasets[dataset_id] = fresh
        _save_dataset_metadata(dataset_id, fresh)
        dataset = fresh

    return {
        "dataset_id": dataset_id,
        "name": dataset["name"],
        "format": dataset["format"],
        "task_type": dataset.get("task_type", stats.get("task_type", "detection")),
        "total_images": stats.get("total_images", dataset["num_images"]),
        "total_annotations": stats.get("total_annotations", dataset["num_annotations"]),
        "class_distribution": stats.get("class_distribution", {}),
        "splits": stats.get("splits", {}),
        "image_sizes": {},
        "avg_annotations_per_image": stats.get("avg_annotations_per_image", 0),
        "created_at": dataset["created_at"]
    }


@app.delete("/api/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset"""
    if dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset_path = DATASETS_DIR / dataset_id
    shutil.rmtree(dataset_path, ignore_errors=True)
    del active_datasets[dataset_id]
    
    return {"success": True}


# ============== DATASET SPLITTING ==============

@app.post("/api/datasets/{dataset_id}/split")
async def split_dataset(dataset_id: str, request: SplitRequest):
    """Split a dataset into train/val/test sets"""
    if dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Validate ratios
    total_ratio = request.train_ratio + request.val_ratio + request.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise HTTPException(status_code=400, detail="Ratios must sum to 1.0")
    
    dataset = active_datasets[dataset_id]
    dataset_path = DATASETS_DIR / dataset_id
    
    # Create new dataset with splits
    new_dataset_id = str(uuid.uuid4())
    output_path = DATASETS_DIR / new_dataset_id
    
    split_result = dataset_parser.create_split_dataset(
        dataset_path,
        output_path,
        dataset["format"],
        train_ratio=request.train_ratio,
        val_ratio=request.val_ratio,
        test_ratio=request.test_ratio,
        shuffle=request.shuffle,
        seed=request.seed
    )
    
    # Parse new dataset
    output_name = request.output_name or f"{dataset['name']}_split"
    new_info = dataset_parser.parse_dataset(output_path, dataset["format"], output_name)
    new_info["id"] = new_dataset_id
    new_info["splits"] = split_result["splits"]
    active_datasets[new_dataset_id] = new_info
    
    return {
        "success": True, 
        "new_dataset": new_info,
        "splits": split_result["splits"]
    }


# ============== CLASS MANAGEMENT ==============

@app.post("/api/datasets/{dataset_id}/extract-classes")
async def extract_classes_to_new_dataset(dataset_id: str, request: ClassExtractRequest):
    """Extract specific classes to a new dataset"""
    if dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = active_datasets[dataset_id]
    dataset_path = DATASETS_DIR / dataset_id
    
    # Validate classes exist
    for cls in request.classes_to_extract:
        if cls not in dataset["classes"]:
            raise HTTPException(status_code=400, detail=f"Class '{cls}' not found in dataset")
    
    # Create new dataset
    new_dataset_id = str(uuid.uuid4())
    output_path = DATASETS_DIR / new_dataset_id
    
    extraction_result = annotation_manager.extract_classes(
        dataset_path,
        output_path,
        dataset["format"],
        request.classes_to_extract
    )
    
    # Parse new dataset
    new_info = dataset_parser.parse_dataset(output_path, dataset["format"], request.output_name)
    new_info["id"] = new_dataset_id
    active_datasets[new_dataset_id] = new_info
    
    return {
        "success": True,
        "new_dataset": new_info,
        "extracted_images": extraction_result["extracted_images"],
        "extracted_annotations": extraction_result["extracted_annotations"]
    }


@app.post("/api/datasets/{dataset_id}/delete-classes")
async def delete_classes_from_dataset(dataset_id: str, request: ClassDeleteRequest):
    """Delete specific classes from a dataset"""
    if dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = active_datasets[dataset_id]
    dataset_path = DATASETS_DIR / dataset_id
    
    # Delete classes
    delete_result = annotation_manager.delete_classes(
        dataset_path,
        dataset["format"],
        request.classes_to_delete
    )
    
    # Update dataset info
    dataset_info = dataset_parser.parse_dataset(dataset_path, dataset["format"], dataset["name"])
    dataset_info["id"] = dataset_id
    active_datasets[dataset_id] = dataset_info
    
    return {
        "success": True,
        "updated_dataset": dataset_info,
        "deleted_annotations": delete_result["deleted_annotations"],
        "affected_images": delete_result["affected_images"]
    }


@app.post("/api/datasets/{dataset_id}/merge-classes")
async def merge_classes_in_dataset(dataset_id: str, request: ClassMergeRequest):
    """Merge multiple classes into one"""
    if dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = active_datasets[dataset_id]
    dataset_path = DATASETS_DIR / dataset_id
    
    # Merge classes
    merge_result = annotation_manager.merge_classes(
        dataset_path,
        dataset["format"],
        request.source_classes,
        request.target_class
    )
    
    # Update dataset info
    dataset_info = dataset_parser.parse_dataset(dataset_path, dataset["format"], dataset["name"])
    dataset_info["id"] = dataset_id
    active_datasets[dataset_id] = dataset_info
    
    return {
        "success": True,
        "updated_dataset": dataset_info,
        "merged_annotations": merge_result["merged_annotations"]
    }


@app.post("/api/datasets/{dataset_id}/add-classes")
async def add_classes_to_dataset(dataset_id: str, request: ClassAddRequest):
    """Add new classes to an existing dataset"""
    if dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = active_datasets[dataset_id]
    dataset_path = DATASETS_DIR / dataset_id
    
    if request.use_model and request.model_id:
        # Auto-annotate with new classes using model
        results = model_manager.annotate_with_new_classes(
            request.model_id,
            dataset_path,
            dataset["format"],
            request.new_classes
        )
    else:
        # Just add classes to the dataset configuration
        annotation_manager.add_classes(
            dataset_path,
            dataset["format"],
            request.new_classes
        )
        results = {"added_classes": request.new_classes}
    
    # Update dataset info
    dataset_info = dataset_parser.parse_dataset(dataset_path, dataset["format"], dataset["name"])
    dataset_info["id"] = dataset_id
    active_datasets[dataset_id] = dataset_info
    
    return {"success": True, "results": results, "updated_dataset": dataset_info}


@app.get("/api/datasets/{dataset_id}/classes")
async def get_dataset_classes(dataset_id: str):
    """Get all classes in a dataset with annotation counts."""
    # Auto-restore if this dataset was not in memory (e.g. after a reload)
    if dataset_id not in active_datasets:
        dataset_path = DATASETS_DIR / dataset_id
        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")
        try:
            meta_path = dataset_path / METADATA_FILENAME
            if meta_path.exists():
                with open(meta_path) as f:
                    info = json.load(f)
                info["id"] = dataset_id
            else:
                info = dataset_parser.parse_dataset(dataset_path, name=dataset_path.name)
                info["id"] = dataset_id
                _save_dataset_metadata(dataset_id, info)
            active_datasets[dataset_id] = info
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not restore dataset: {e}")

    dataset = active_datasets[dataset_id]
    dataset_path = DATASETS_DIR / dataset_id

    classes_list = dataset_parser.get_classes_with_distribution(dataset_path, dataset["format"])
    # Return as a plain dict {name: count} — simpler for frontend to consume
    classes_dict = {item["name"]: item["count"] for item in classes_list}
    return {"classes": classes_dict}


# ============== AUGMENTATION ==============

@app.get("/api/augmentations")
async def list_available_augmentations():
    """List all available augmentation options"""
    return {
        "augmentations": [
            {
                "id": "flip_horizontal",
                "name": "Horizontal Flip",
                "description": "Flip images horizontally",
                "params": {}
            },
            {
                "id": "flip_vertical",
                "name": "Vertical Flip",
                "description": "Flip images vertically",
                "params": {}
            },
            {
                "id": "rotate",
                "name": "Rotation",
                "description": "Rotate images by a random angle",
                "params": {"angle": {"type": "range", "min": -45, "max": 45, "default": 15}}
            },
            {
                "id": "brightness",
                "name": "Brightness",
                "description": "Adjust image brightness",
                "params": {"factor": {"type": "range", "min": 0.5, "max": 1.5, "default": 0.2}}
            },
            {
                "id": "contrast",
                "name": "Contrast",
                "description": "Adjust image contrast",
                "params": {"factor": {"type": "range", "min": 0.5, "max": 1.5, "default": 0.2}}
            },
            {
                "id": "saturation",
                "name": "Saturation",
                "description": "Adjust color saturation",
                "params": {"factor": {"type": "range", "min": 0.5, "max": 1.5, "default": 0.2}}
            },
            {
                "id": "hue",
                "name": "Hue Shift",
                "description": "Shift image hue",
                "params": {"factor": {"type": "range", "min": -0.1, "max": 0.1, "default": 0.05}}
            },
            {
                "id": "blur",
                "name": "Gaussian Blur",
                "description": "Apply Gaussian blur",
                "params": {"kernel": {"type": "range", "min": 3, "max": 11, "default": 5}}
            },
            {
                "id": "noise",
                "name": "Gaussian Noise",
                "description": "Add random noise",
                "params": {"variance": {"type": "range", "min": 0.01, "max": 0.1, "default": 0.02}}
            },
            {
                "id": "crop",
                "name": "Random Crop",
                "description": "Randomly crop a portion of the image",
                "params": {"scale": {"type": "range", "min": 0.7, "max": 0.95, "default": 0.85}}
            },
            {
                "id": "scale",
                "name": "Random Scale",
                "description": "Randomly scale the image",
                "params": {"scale": {"type": "range", "min": 0.8, "max": 1.2, "default": 0.1}}
            },
            {
                "id": "shear",
                "name": "Shear",
                "description": "Apply shear transformation",
                "params": {"angle": {"type": "range", "min": -15, "max": 15, "default": 10}}
            },
            {
                "id": "mosaic",
                "name": "Mosaic",
                "description": "Combine 4 images into one",
                "params": {}
            },
            {
                "id": "mixup",
                "name": "MixUp",
                "description": "Blend two images together",
                "params": {"alpha": {"type": "range", "min": 0.1, "max": 0.5, "default": 0.3}}
            },
            {
                "id": "cutout",
                "name": "Cutout",
                "description": "Randomly cut out rectangles",
                "params": {"num_holes": {"type": "range", "min": 1, "max": 5, "default": 2}, "size": {"type": "range", "min": 10, "max": 50, "default": 30}}
            },
            {
                "id": "grayscale",
                "name": "Grayscale",
                "description": "Convert to grayscale",
                "params": {"probability": {"type": "range", "min": 0, "max": 1, "default": 0.1}}
            },
            {
                "id": "elastic",
                "name": "Elastic Deformation",
                "description": "Apply elastic transformation",
                "params": {"alpha": {"type": "range", "min": 50, "max": 200, "default": 100}, "sigma": {"type": "range", "min": 5, "max": 15, "default": 10}}
            }
        ]
    }


@app.post("/api/datasets/{dataset_id}/augment")
async def augment_dataset(dataset_id: str, config: AugmentationConfig, background_tasks: BackgroundTasks):
    """Augment a dataset to reach target size"""
    if dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = active_datasets[dataset_id]
    dataset_path = DATASETS_DIR / dataset_id
    
    # Create new augmented dataset
    new_dataset_id = str(uuid.uuid4())
    output_path = DATASETS_DIR / new_dataset_id
    
    augmentation_result = annotation_manager.augment_dataset(
        dataset_path,
        output_path,
        dataset["format"],
        config.target_size,
        config.augmentations
    )
    
    # Parse new dataset
    new_info = dataset_parser.parse_dataset(output_path, dataset["format"], config.output_name)
    new_info["id"] = new_dataset_id
    active_datasets[new_dataset_id] = new_info
    
    return {
        "success": True,
        "new_dataset": new_info,
        "augmented_images": augmentation_result["augmented_images"],
        "original_images": augmentation_result["original_images"]
    }


# ============== ANNOTATION HISTORY ==============

@app.get("/api/datasets/{dataset_id}/history")
async def get_annotation_history(dataset_id: str, image_id: str = None):
    """Get annotation history for a dataset or specific image"""
    if dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    history_key = f"{dataset_id}_{image_id}" if image_id else dataset_id
    history = annotation_history.get(history_key, [])
    
    return {"history": history}


@app.post("/api/datasets/{dataset_id}/history/undo")
async def undo_annotation(dataset_id: str, image_id: str):
    """Undo the last annotation action"""
    if dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    history_key = f"{dataset_id}_{image_id}"
    history = annotation_history.get(history_key, [])
    
    if not history:
        raise HTTPException(status_code=400, detail="No history to undo")
    
    # Get the last entry and restore previous state
    last_entry = history.pop()
    
    dataset = active_datasets[dataset_id]
    dataset_path = DATASETS_DIR / dataset_id
    
    # Restore annotations
    if last_entry.get("previous_annotations"):
        annotation_manager.restore_annotations(
            dataset_path,
            dataset["format"],
            image_id,
            last_entry["previous_annotations"]
        )
    
    annotation_history[history_key] = history
    
    return {"success": True, "restored_to": last_entry.get("previous_annotations")}


@app.post("/api/datasets/{dataset_id}/history/record")
async def record_annotation_action(dataset_id: str, image_id: str, action: str, previous_annotations: List[Dict], new_annotations: List[Dict]):
    """Record an annotation action for undo functionality"""
    if dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    history_key = f"{dataset_id}_{image_id}"
    if history_key not in annotation_history:
        annotation_history[history_key] = []
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "previous_annotations": previous_annotations,
        "new_annotations": new_annotations
    }
    
    # Keep only last 50 entries per image
    annotation_history[history_key].append(entry)
    if len(annotation_history[history_key]) > 50:
        annotation_history[history_key] = annotation_history[history_key][-50:]
    
    return {"success": True}


# ============== DATASET SORTING ==============

@app.post("/api/sorting/start/{dataset_id}")
async def start_sorting_session(dataset_id: str, filter_classes: List[str] = Query(default=None)):
    """Start a sorting/filtering session for a dataset"""
    if dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = active_datasets[dataset_id]
    dataset_path = DATASETS_DIR / dataset_id
    
    # Get all images with annotations, optionally filtered by class
    images = dataset_parser.get_images_with_annotations(
        dataset_path, 
        dataset["format"],
        filter_classes=filter_classes
    )
    
    session_id = str(uuid.uuid4())
    sorting_sessions[session_id] = {
        "dataset_id": dataset_id,
        "images": images,
        "current_index": 0,
        "kept": [],
        "deleted": [],
        "total": len(images),
        "filter_classes": filter_classes
    }
    
    return {
        "session_id": session_id,
        "total_images": len(images),
        "current_image": images[0] if images else None
    }


@app.get("/api/sorting/{session_id}/current")
async def get_current_sorting_image(session_id: str):
    """Get the current image in sorting session"""
    if session_id not in sorting_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sorting_sessions[session_id]
    idx = session["current_index"]
    
    if idx >= len(session["images"]):
        return {"complete": True, "kept": len(session["kept"]), "deleted": len(session["deleted"])}
    
    return {
        "complete": False,
        "current_index": idx,
        "total": session["total"],
        "image": session["images"][idx],
        "progress": {
            "kept": len(session["kept"]),
            "deleted": len(session["deleted"]),
            "remaining": session["total"] - idx
        }
    }


@app.post("/api/sorting/{session_id}/action")
async def sorting_action(session_id: str, action: SortingAction):
    """Process a sorting action (keep/delete) for current image"""
    if session_id not in sorting_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sorting_sessions[session_id]
    idx = session["current_index"]
    
    if idx >= len(session["images"]):
        return {"complete": True}
    
    image = session["images"][idx]
    
    if action.action == "keep":
        session["kept"].append(image)
    elif action.action == "delete":
        session["deleted"].append(image)
    
    session["current_index"] += 1
    
    # Return next image
    if session["current_index"] >= len(session["images"]):
        return {"complete": True, "kept": len(session["kept"]), "deleted": len(session["deleted"])}
    
    return {
        "complete": False,
        "current_index": session["current_index"],
        "image": session["images"][session["current_index"]],
        "progress": {
            "kept": len(session["kept"]),
            "deleted": len(session["deleted"]),
            "remaining": session["total"] - session["current_index"]
        }
    }


@app.post("/api/sorting/{session_id}/go-back")
async def go_back_in_sorting(session_id: str):
    """Go back to the previous image in sorting"""
    if session_id not in sorting_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sorting_sessions[session_id]
    
    if session["current_index"] <= 0:
        raise HTTPException(status_code=400, detail="Already at first image")
    
    session["current_index"] -= 1
    
    # Remove from kept/deleted lists
    current_image_id = session["images"][session["current_index"]]["id"]
    session["kept"] = [img for img in session["kept"] if img["id"] != current_image_id]
    session["deleted"] = [img for img in session["deleted"] if img["id"] != current_image_id]
    
    return {
        "complete": False,
        "current_index": session["current_index"],
        "image": session["images"][session["current_index"]],
        "progress": {
            "kept": len(session["kept"]),
            "deleted": len(session["deleted"]),
            "remaining": session["total"] - session["current_index"]
        }
    }


@app.post("/api/sorting/{session_id}/finalize")
async def finalize_sorting(session_id: str, create_new_dataset: bool = True):
    """Finalize sorting session and optionally create new filtered dataset"""
    if session_id not in sorting_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sorting_sessions[session_id]
    dataset_id = session["dataset_id"]
    
    if create_new_dataset:
        # Create new dataset with only kept images
        new_dataset_id = str(uuid.uuid4())
        new_dataset_path = DATASETS_DIR / new_dataset_id
        
        original_path = DATASETS_DIR / dataset_id
        original_format = active_datasets[dataset_id]["format"]
        
        dataset_parser.create_filtered_dataset(
            original_path,
            new_dataset_path,
            session["kept"],
            original_format
        )
        
        # Parse new dataset
        new_info = dataset_parser.parse_dataset(
            new_dataset_path,
            format_hint=original_format,
            name=f"{active_datasets[dataset_id]['name']}_filtered"
        )
        new_info["id"] = new_dataset_id
        active_datasets[new_dataset_id] = new_info
        
        del sorting_sessions[session_id]
        return {"success": True, "new_dataset": new_info}
    
    del sorting_sessions[session_id]
    return {"success": True}


# ============== ANNOTATION TOOLS ==============

@app.get("/api/datasets/{dataset_id}/images")
async def get_dataset_images(
    dataset_id: str,
    page: int = 1,
    limit: int = 50,
    split: str = None,
    class_name: str = None,
    bust_cache: bool = False
):
    """Get paginated list of images in dataset with optional split/class filtering"""
    if dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = active_datasets[dataset_id]
    dataset_path = DATASETS_DIR / dataset_id

    # Use cache to avoid re-parsing large datasets on every request
    if bust_cache or dataset_id not in _images_cache:
        all_images = dataset_parser.get_images_with_annotations(
            dataset_path,
            dataset["format"],
            page=1,
            limit=999999
        )
        _images_cache[dataset_id] = all_images
    else:
        all_images = _images_cache[dataset_id]

    # Filter by split
    if split and split != "all":
        all_images = [img for img in all_images if img.get("split") == split]

    # Filter by class
    if class_name and class_name != "all":
        filtered = []
        for img in all_images:
            if img.get("class_name") == class_name:
                filtered.append(img)
            elif any(ann.get("class_name") == class_name for ann in img.get("annotations", [])):
                filtered.append(img)
        all_images = filtered

    # Paginate (if limit is very large, return all)
    if limit >= 999999:
        images = all_images
    else:
        start = (page - 1) * limit
        images = all_images[start:start + limit]

    return {"images": images, "total": len(all_images), "page": page, "limit": limit}


@app.get("/api/datasets/{dataset_id}/image/{image_id}")
async def get_image_with_annotations(dataset_id: str, image_id: str):
    """Get a specific image with its annotations"""
    if dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = active_datasets[dataset_id]
    dataset_path = DATASETS_DIR / dataset_id
    
    image_data = dataset_parser.get_image_data(dataset_path, dataset["format"], image_id)
    return image_data


@app.put("/api/datasets/{dataset_id}/image/{image_id}/annotations")
async def update_annotations(dataset_id: str, image_id: str, update: AnnotationUpdate):
    """Update annotations for an image"""
    if dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = active_datasets[dataset_id]
    dataset_path = DATASETS_DIR / dataset_id
    
    annotation_manager.update_annotations(
        dataset_path,
        dataset["format"],
        image_id,
        update.annotations
    )
    
    # Invalidate image cache so next fetch reflects new annotations
    if dataset_id in _images_cache:
        # Update just the one image in cache rather than flushing entire cache
        cache = _images_cache[dataset_id]
        for img in cache:
            if img.get("id") == image_id:
                img["annotations"] = [a.dict() if hasattr(a, "dict") else a for a in update.annotations]
                img["has_annotations"] = len(update.annotations) > 0
                break

    # Update dataset stats
    dataset_info = dataset_parser.parse_dataset(dataset_path, dataset["format"], dataset["name"])
    dataset_info["id"] = dataset_id
    active_datasets[dataset_id] = dataset_info
    
    return {"success": True}


@app.post("/api/datasets/{dataset_id}/add-images")
async def add_images_to_dataset(dataset_id: str, files: List[UploadFile] = File(...)):
    """Add new unannotated images to a dataset"""
    if dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = active_datasets[dataset_id]
    dataset_path = DATASETS_DIR / dataset_id
    
    added_images = []
    for file in files:
        if file.content_type.startswith("image/"):
            image_path = annotation_manager.add_image(dataset_path, dataset["format"], file)
            added_images.append(image_path)
    
    # Update dataset info
    dataset_info = dataset_parser.parse_dataset(dataset_path, dataset["format"], dataset["name"])
    dataset_info["id"] = dataset_id
    active_datasets[dataset_id] = dataset_info
    
    return {"success": True, "added_images": added_images}


# ============== AUTO-ANNOTATION ==============

@app.get("/api/models")
async def list_models():
    """List available models for auto-annotation"""
    return {"models": model_manager.list_models()}


@app.post("/api/models/load")
async def load_model(
    model_file: UploadFile = File(None),
    model_type: str = "yolo",
    model_name: str = None,
    pretrained: str = None
):
    """Load a model for auto-annotation"""
    try:
        if model_file:
            # Save uploaded model
            model_path = MODELS_DIR / (model_name or model_file.filename)
            content = await model_file.read()
            with open(model_path, "wb") as f:
                f.write(content)
            
            model_id = model_manager.load_model(str(model_path), model_type)
        else:
            # Load pretrained model
            model_id = model_manager.load_pretrained(model_type, pretrained)
        
        return {"success": True, "model_id": model_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/models/download")
async def start_model_download(
    background_tasks: BackgroundTasks,
    model_type: str = Form(...),
    pretrained: str = Form(...),
    hf_token: Optional[str] = Form(None),
):
    """Start a background download for a pretrained model."""
    if _download_status.get(pretrained, {}).get("status") == "downloading":
        return {"success": True, "already_started": True}

    _download_status[pretrained] = {"status": "downloading", "progress": 5}

    def _do_download(mtype: str, mname: str, token: Optional[str]):
        try:
            _download_status[mname]["progress"] = 15
            model_manager.load_pretrained(mtype, mname, hf_token=token)
            info = model_manager.loaded_models.get(mname, {})
            if info.get("error") and not info.get("path"):
                _download_status[mname] = {"status": "error", "progress": 0, "error": info["error"]}
            else:
                _download_status[mname] = {"status": "done", "progress": 100}
        except Exception as e:
            _download_status[mname] = {"status": "error", "progress": 0, "error": str(e)}

    background_tasks.add_task(_do_download, model_type, pretrained, hf_token)
    return {"success": True, "started": True}


@app.get("/api/models/download-status/{model_id}")
async def get_download_status(model_id: str):
    """Poll the progress of an ongoing model download."""
    return _download_status.get(model_id, {"status": "idle", "progress": 0})


@app.post("/api/auto-annotate/{dataset_id}")
async def auto_annotate_dataset(
    dataset_id: str,
    model_id: str,
    confidence_threshold: float = 0.5,
    background_tasks: BackgroundTasks = None
):
    """Auto-annotate a dataset using a loaded model"""
    if dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = active_datasets[dataset_id]
    dataset_path = DATASETS_DIR / dataset_id
    
    # Run auto-annotation
    results = model_manager.auto_annotate(
        model_id,
        dataset_path,
        dataset["format"],
        confidence_threshold
    )
    
    # Update dataset info
    dataset_info = dataset_parser.parse_dataset(dataset_path, dataset["format"], dataset["name"])
    dataset_info["id"] = dataset_id
    active_datasets[dataset_id] = dataset_info
    
    return {"success": True, "annotated_images": results["annotated_count"]}


@app.post("/api/auto-annotate/{dataset_id}/single/{image_id}")
async def auto_annotate_single_image(
    dataset_id: str,
    image_id: str,
    model_id: str,
    confidence_threshold: float = 0.5,
    point_x: Optional[float] = None,
    point_y: Optional[float] = None,
    text_prompt: Optional[str] = None,
    image_path_hint: Optional[str] = None,
):
    """Auto-annotate a single image, with optional SAM click-point or text prompt."""
    if dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = active_datasets[dataset_id]
    dataset_path = DATASETS_DIR / dataset_id

    prompt_point = (point_x, point_y) if point_x is not None and point_y is not None else None

    annotations = model_manager.annotate_single_image(
        model_id,
        dataset_path,
        dataset["format"],
        image_id,
        confidence_threshold,
        prompt_point=prompt_point,
        text_prompt=text_prompt or None,
        image_path_hint=image_path_hint,
    )

    return {"success": True, "annotations": annotations}


# Batch text-annotation job tracking
_text_annotate_jobs: Dict[str, Dict] = {}
# Per-job threading controls  {job_id: {"pause": Event (set=run, clear=pause), "stop": Event}}
_job_controls: Dict[str, Dict] = {}


def _persist_jobs():
    """Persist batch job state to disk so it survives server restarts."""
    try:
        with open(JOBS_FILE, "w") as f:
            json.dump(_text_annotate_jobs, f, indent=2, default=str)
    except Exception:
        pass


def _restore_jobs():
    """On startup, restore persisted batch jobs from disk.
    Any jobs that were running or paused are marked 'interrupted' since
    the background threads are gone after a restart."""
    if not JOBS_FILE.exists():
        return
    try:
        with open(JOBS_FILE) as f:
            jobs = json.load(f)
        count = 0
        for job_id, job in jobs.items():
            if job.get("status") in ("running", "paused"):
                job["status"] = "interrupted"
                job["paused"] = False
            _text_annotate_jobs[job_id] = job
            count += 1
        if count:
            print(f"[startup] Restored {count} batch job(s) from disk.")
    except Exception:
        pass

@app.post("/api/auto-annotate/{dataset_id}/text-batch")
async def auto_annotate_text_batch(
    dataset_id: str,
    model_id: str,
    text_prompt: str,
    confidence_threshold: float = 0.5,
    background_tasks: BackgroundTasks = None,
):
    """
    Batch-annotate every image in a dataset using a SAM3 text prompt.
    Runs in the background; poll /api/auto-annotate/text-batch/{job_id}/status.
    """
    if dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = active_datasets[dataset_id]
    dataset_path = DATASETS_DIR / dataset_id
    job_id = str(uuid.uuid4())[:8]

    import threading as _threading
    pause_ev = _threading.Event(); pause_ev.set()   # set = running, clear = paused
    stop_ev  = _threading.Event()                   # set = stop requested

    _text_annotate_jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "paused": False,
        "progress": 0,
        "total": 0,
        "processed": 0,          # images attempted (annotated + failed)
        "annotated": 0,          # images with ≥1 annotation saved
        "failed": 0,
        "total_annotations": 0,  # total annotation objects created
        "dataset_id": dataset_id,
        "text_prompt": text_prompt,
        "started_at": datetime.utcnow().isoformat(),
        "recent_images": [],     # last 10 processed images for live preview
    }
    _job_controls[job_id] = {"pause": pause_ev, "stop": stop_ev}
    _persist_jobs()

    def _run_batch(job_id: str):
        try:
            # Collect all images in the dataset
            IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
            image_files = []
            for ext in IMAGE_EXTENSIONS:
                image_files.extend(dataset_path.rglob(f"*{ext}"))
                image_files.extend(dataset_path.rglob(f"*{ext.upper()}"))
            # Exclude images inside workspace/models or similar non-dataset dirs
            image_files = [f for f in image_files if "labels" not in f.parts]

            _text_annotate_jobs[job_id]["total"] = len(image_files)
            classes = model_manager._get_dataset_classes(dataset_path, dataset["format"])

            ctrl = _job_controls[job_id]
            for i, image_file in enumerate(image_files):
                # ── Stop check ──────────────────────────────────────────────────
                if ctrl["stop"].is_set():
                    _text_annotate_jobs[job_id]["status"] = "cancelled"
                    _images_cache.pop(dataset_id, None)
                    return

                # ── Pause: block until resumed or stopped ────────────────────
                if not ctrl["pause"].is_set():
                    _text_annotate_jobs[job_id]["paused"] = True
                    _text_annotate_jobs[job_id]["status"] = "paused"
                    ctrl["pause"].wait()           # blocks here
                    if ctrl["stop"].is_set():
                        _text_annotate_jobs[job_id]["status"] = "cancelled"
                        _images_cache.pop(dataset_id, None)
                        return
                    _text_annotate_jobs[job_id]["paused"] = False
                    _text_annotate_jobs[job_id]["status"] = "running"

                try:
                    annotations = model_manager._run_inference(
                        model_manager.loaded_models[model_id]["model"],
                        model_manager.loaded_models[model_id]["type"],
                        image_file,
                        confidence_threshold,
                        text_prompt=text_prompt,
                    )
                except Exception as exc:
                    print(f"[batch:{job_id}] inference failed on {image_file.name}: {exc}")
                    _text_annotate_jobs[job_id]["failed"] += 1
                    _text_annotate_jobs[job_id]["processed"] = i + 1
                    _text_annotate_jobs[job_id]["progress"] = int((i + 1) / max(len(image_files), 1) * 100)
                    continue

                if annotations:
                    try:
                        model_manager._save_annotations(
                            dataset_path, dataset["format"], image_file, annotations, classes
                        )
                        _text_annotate_jobs[job_id]["annotated"] += 1
                        _text_annotate_jobs[job_id]["total_annotations"] += len(annotations)
                        # Track recent images for live preview in the UI
                        try:
                            rel_path = str(image_file.relative_to(dataset_path))
                        except Exception:
                            rel_path = image_file.name
                        recent = _text_annotate_jobs[job_id].get("recent_images", [])
                        recent.append({
                            "filename": image_file.name,
                            "path": rel_path,
                            "abs_path": str(image_file),
                            "image_id": image_file.stem,
                            "annotation_count": len(annotations),
                        })
                        _text_annotate_jobs[job_id]["recent_images"] = recent[-10:]
                        # Keep image list cache fresh so live polling returns new annotations
                        _images_cache.pop(dataset_id, None)
                    except Exception as exc:
                        print(f"[batch:{job_id}] save failed on {image_file.name}: {exc}")
                        _text_annotate_jobs[job_id]["failed"] += 1
                _text_annotate_jobs[job_id]["processed"] = i + 1
                _text_annotate_jobs[job_id]["progress"] = int((i + 1) / max(len(image_files), 1) * 100)
                # Persist state every 5 images
                if i % 5 == 4:
                    _persist_jobs()

            # Refresh cache
            _images_cache.pop(dataset_id, None)
            _text_annotate_jobs[job_id]["status"] = "done"
            _persist_jobs()
        except Exception as e:
            _text_annotate_jobs[job_id]["status"] = "error"
            _text_annotate_jobs[job_id]["error"] = str(e)
            _persist_jobs()

    # Auto-load model if not in memory
    if model_id not in model_manager.loaded_models or not model_manager.loaded_models[model_id].get("model"):
        model_manager._try_autoload(model_id)

    if model_id not in model_manager.loaded_models or not model_manager.loaded_models[model_id].get("model"):
        raise HTTPException(status_code=400, detail="Model not loaded or unavailable")

    import threading
    threading.Thread(target=_run_batch, args=(job_id,), daemon=True).start()

    return {"job_id": job_id, "status": "running"}


@app.get("/api/auto-annotate/text-batch/{job_id}/status")
async def get_text_batch_status(job_id: str):
    """Poll the status of a running text-batch annotation job."""
    if job_id not in _text_annotate_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _text_annotate_jobs[job_id]


@app.post("/api/auto-annotate/text-batch/{job_id}/pause")
async def pause_text_batch(job_id: str):
    """Pause a running batch job between images."""
    if job_id not in _job_controls:
        raise HTTPException(status_code=404, detail="Job not found")
    if _text_annotate_jobs[job_id]["status"] not in ("running",):
        raise HTTPException(status_code=400, detail="Job is not running")
    _job_controls[job_id]["pause"].clear()   # clear = blocked/paused
    _text_annotate_jobs[job_id]["paused"] = True
    _text_annotate_jobs[job_id]["status"] = "paused"
    _persist_jobs()
    return {"status": "paused"}


@app.post("/api/auto-annotate/text-batch/{job_id}/resume")
async def resume_text_batch(job_id: str):
    """Resume a paused batch job."""
    if job_id not in _job_controls:
        raise HTTPException(status_code=404, detail="Job not found")
    if _text_annotate_jobs[job_id]["status"] not in ("paused",):
        raise HTTPException(status_code=400, detail="Job is not paused")
    _text_annotate_jobs[job_id]["paused"] = False
    _text_annotate_jobs[job_id]["status"] = "running"
    _job_controls[job_id]["pause"].set()    # set = unblocked/running
    _persist_jobs()
    return {"status": "running"}


@app.post("/api/auto-annotate/text-batch/{job_id}/cancel")
async def cancel_text_batch(job_id: str):
    """Cancel a running or paused batch job."""
    if job_id not in _text_annotate_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    current_status = _text_annotate_jobs[job_id]["status"]
    if current_status in ("done", "cancelled", "error"):
        raise HTTPException(status_code=400, detail="Job is already finished")
    # Signal the background thread if it's still alive
    if job_id in _job_controls:
        ctrl = _job_controls[job_id]
        ctrl["stop"].set()
        ctrl["pause"].set()   # unblock a paused thread so it can see stop_ev
    _text_annotate_jobs[job_id]["status"] = "cancelled"
    _persist_jobs()
    return {"status": "cancelled"}


@app.get("/api/auto-annotate/jobs")
async def list_batch_jobs():
    """Return all known batch jobs (for frontend state restoration)."""
    return {"jobs": list(_text_annotate_jobs.values())}


@app.get("/api/auto-annotate/text-batch/{job_id}/preview/{idx}")
async def get_batch_job_preview_image(job_id: str, idx: int):
    """Serve a recently-processed image from a batch job for live preview."""
    if job_id not in _text_annotate_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    recent = _text_annotate_jobs[job_id].get("recent_images", [])
    if idx < 0 or idx >= len(recent):
        raise HTTPException(status_code=404, detail="Image index out of range")
    abs_path = recent[idx].get("abs_path")
    if not abs_path or not Path(abs_path).exists():
        raise HTTPException(status_code=404, detail="Image file not found")
    return FileResponse(abs_path)


# ============== FORMAT CONVERSION ==============

@app.get("/api/formats")
async def list_formats():
    """List all supported annotation formats"""
    return {
        "formats": [
            {"id": "yolo", "name": "YOLO", "description": "YOLOv5/v8 format with txt annotations", "tasks": ["detection", "segmentation"]},
            {"id": "coco", "name": "COCO", "description": "COCO JSON format", "tasks": ["detection", "segmentation", "keypoints"]},
            {"id": "voc", "name": "Pascal VOC", "description": "XML annotations", "tasks": ["detection"]},
            {"id": "labelme", "name": "LabelMe", "description": "LabelMe JSON format", "tasks": ["detection", "segmentation"]},
            {"id": "createml", "name": "CreateML", "description": "Apple CreateML format", "tasks": ["detection", "classification"]},
            {"id": "tfrecord", "name": "TFRecord", "description": "TensorFlow Record format", "tasks": ["detection", "classification"]},
            {"id": "csv", "name": "CSV", "description": "Simple CSV format", "tasks": ["detection", "classification"]},
            {"id": "yolo_seg", "name": "YOLO Segmentation", "description": "YOLO polygon segmentation", "tasks": ["segmentation"]},
            {"id": "coco_panoptic", "name": "COCO Panoptic", "description": "COCO panoptic segmentation", "tasks": ["segmentation"]},
            {"id": "cityscapes", "name": "Cityscapes", "description": "Cityscapes format", "tasks": ["segmentation"]},
            {"id": "ade20k", "name": "ADE20K", "description": "ADE20K segmentation format", "tasks": ["segmentation"]},
            {"id": "classification", "name": "Classification Folder", "description": "Folder structure for classification", "tasks": ["classification"]},
            {"id": "yolo_obb", "name": "YOLO OBB", "description": "YOLO oriented bounding boxes", "tasks": ["detection"]},
            {"id": "dota", "name": "DOTA", "description": "DOTA format for oriented objects", "tasks": ["detection"]}
        ]
    }


@app.post("/api/convert")
async def convert_dataset(request: ConversionRequest):
    """Convert a dataset to a different format"""
    if request.dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = active_datasets[request.dataset_id]
    source_path = DATASETS_DIR / request.dataset_id
    
    # Map frontend format names to backend format names
    format_map = {
        "csv": "tensorflow-csv",
        "tensorflow-csv": "tensorflow-csv",
        "voc": "pascal-voc",
        "pascal-voc": "pascal-voc",
        "yolo_seg": "yolo",
        "yolo-seg": "yolo",
    }
    target_format = format_map.get(request.target_format, request.target_format)
    
    # Create new dataset for converted version
    new_dataset_id = str(uuid.uuid4())
    output_path = DATASETS_DIR / new_dataset_id
    
    try:
        format_converter.convert(
            source_path,
            output_path,
            dataset["format"],
            target_format
        )
    except Exception as e:
        shutil.rmtree(output_path, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Conversion failed: {str(e)}")
    
    # Parse new dataset
    output_name = request.output_name or f"{dataset['name']}_{request.target_format}"
    new_info = dataset_parser.parse_dataset(output_path, target_format, output_name)
    new_info["id"] = new_dataset_id
    new_info["format"] = request.target_format  # Keep original format name for display
    active_datasets[new_dataset_id] = new_info
    _save_dataset_metadata(new_dataset_id, new_info)
    
    return {"success": True, "new_dataset": new_info}


@app.get("/api/export/{dataset_id}")
async def export_dataset(dataset_id: str, target_format: str = None):
    """Export a dataset as a downloadable zip file"""
    if dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = active_datasets[dataset_id]
    dataset_path = DATASETS_DIR / dataset_id
    
    # Convert if needed
    if target_format and target_format != dataset["format"]:
        export_path = TEMP_DIR / str(uuid.uuid4())
        format_converter.convert(dataset_path, export_path, dataset["format"], target_format)
    else:
        export_path = dataset_path
    
    # Create zip
    zip_path = EXPORTS_DIR / f"{dataset['name']}_{target_format or dataset['format']}.zip"
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", export_path)
    
    return FileResponse(
        zip_path,
        filename=zip_path.name,
        media_type="application/zip"
    )


# ============== DATASET MERGING ==============

@app.post("/api/merge")
async def merge_datasets(request: MergeRequest):
    """Merge multiple datasets into one"""
    for dataset_id in request.dataset_ids:
        if dataset_id not in active_datasets:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    
    # Get dataset paths and formats
    datasets = [
        {
            "path": DATASETS_DIR / did,
            "format": active_datasets[did]["format"],
            "info": active_datasets[did]
        }
        for did in request.dataset_ids
    ]
    
    # Create merged dataset
    new_dataset_id = str(uuid.uuid4())
    output_path = DATASETS_DIR / new_dataset_id
    
    dataset_merger.merge(
        datasets, 
        output_path, 
        request.output_format,
        class_mapping=request.class_mapping
    )
    
    # Parse merged dataset
    new_info = dataset_parser.parse_dataset(output_path, request.output_format, request.output_name)
    new_info["id"] = new_dataset_id
    active_datasets[new_dataset_id] = new_info
    
    return {"success": True, "merged_dataset": new_info}


# ============== TRAINING ==============

def _resolve_base_model(model_arch: str, model_type: str) -> str:
    """Return the correct model identifier for the chosen arch + task type."""
    arch = model_arch.strip()
    # RF-DETR arches are passed through as-is (no .pt extension)
    if model_type == "rfdetr":
        return arch  # e.g. "rfdetr_base" or "rfdetr_large"
    # Strip any existing suffix variants so we can reattach cleanly
    base = arch.replace("-seg", "").replace("-cls", "").replace(".pt", "")
    if model_type == "segmentation":
        filename = f"{base}-seg.pt"
    elif model_type == "classification":
        filename = f"{base}-cls.pt"
    else:
        filename = f"{base}.pt"
    return filename


@app.post("/api/install-cuda-torch")
async def install_cuda_torch():
    """Install CUDA-enabled PyTorch (requires restart after install)."""
    import sys, subprocess
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision",
                "--index-url", "https://download.pytorch.org/whl/cu121",
                "--upgrade",
            ],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            return {"success": True, "message": "CUDA PyTorch installed. Restart the backend server to use GPU."}
        else:
            return {"success": False, "message": result.stderr[-2000:] or result.stdout[-2000:]}
    except subprocess.TimeoutExpired:
        return {"success": False, "message": "Install timed out after 5 minutes."}
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.post("/api/train")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """Start training a model on a dataset"""
    if config.dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = active_datasets[config.dataset_id]
    dataset_path = DATASETS_DIR / config.dataset_id
    
    # Start training in background
    training_id = training_manager.start_training(
        dataset_path,
        dataset["format"],
        config.model_type,
        {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "image_size": config.image_size,
            "pretrained": config.pretrained,
            "device": config.device,
            "base_model": _resolve_base_model(config.model_arch, config.model_type),
            # Advanced hyperparams forwarded for all model types
            "lr0": config.lr0,
            "lrf": config.lrf,
            "optimizer": config.optimizer,
            "patience": config.patience,
            "cos_lr": config.cos_lr,
            "warmup_epochs": config.warmup_epochs,
            "weight_decay": config.weight_decay,
            "mosaic": config.mosaic,
            "hsv_h": config.hsv_h,
            "hsv_s": config.hsv_s,
            "hsv_v": config.hsv_v,
            "flipud": config.flipud,
            "fliplr": config.fliplr,
            "amp": config.amp,
            "dropout": config.dropout,
        }
    )
    
    return {"success": True, "training_id": training_id}


@app.get("/api/train/{training_id}/status")
async def get_training_status(training_id: str):
    """Get training status and metrics"""
    status = training_manager.get_status(training_id)
    if not status:
        raise HTTPException(status_code=404, detail="Training not found")
    return status


@app.post("/api/train/{training_id}/stop")
async def stop_training(training_id: str):
    """Stop an ongoing training"""
    success = training_manager.stop_training(training_id)
    return {"success": success}


@app.get("/api/train/{training_id}/export")
async def export_trained_model(training_id: str):
    """Export a trained model"""
    model_path = training_manager.get_model_path(training_id)
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    
    return FileResponse(model_path, filename=os.path.basename(model_path))


# ============== STATIC FILES ==============

@app.get("/api/image/{dataset_id}/{image_path:path}")
async def serve_image(dataset_id: str, image_path: str):
    """Serve an image from a dataset"""
    if dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_root = dataset_parser._find_dataset_root(DATASETS_DIR / dataset_id)
    full_path = dataset_root / image_path
    if not full_path.exists():
        full_path = DATASETS_DIR / dataset_id / image_path
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(full_path, headers={"Access-Control-Allow-Origin": "*"})


# ============== LOCAL FOLDER MODE ==============

class LocalFolderRequest(BaseModel):
    folder_path: str
    dataset_name: Optional[str] = None
    format_hint: Optional[str] = None


@app.post("/api/datasets/load-local")
async def load_local_dataset(request: LocalFolderRequest):
    """Load a dataset from a local folder path (no upload required)"""
    folder_path = Path(request.folder_path)
    
    if not folder_path.exists():
        raise HTTPException(status_code=404, detail=f"Folder not found: {request.folder_path}")
    
    if not folder_path.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")
    
    dataset_id = str(uuid.uuid4())
    
    # Create symlink or copy based on preference
    dataset_link = DATASETS_DIR / dataset_id
    
    try:
        # Create symlink to avoid copying large datasets
        dataset_link.symlink_to(folder_path.absolute())
    except OSError:
        # If symlink fails (e.g., on Windows), copy the dataset
        shutil.copytree(folder_path, dataset_link)
    
    try:
        # Parse dataset
        dataset_info = dataset_parser.parse_dataset(
            dataset_link,
            format_hint=request.format_hint,
            name=request.dataset_name or folder_path.name
        )
        
        dataset_info["id"] = dataset_id
        dataset_info["local_path"] = str(folder_path.absolute())
        dataset_info["is_local"] = True
        active_datasets[dataset_id] = dataset_info
        _save_dataset_metadata(dataset_id, dataset_info)
        
        return {"success": True, "dataset": dataset_info}
    
    except Exception as e:
        # Cleanup on failure
        if dataset_link.is_symlink():
            dataset_link.unlink()
        else:
            shutil.rmtree(dataset_link, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/datasets/browse-folders")
async def browse_local_folders(path: str = "."):
    """Browse local folders to find datasets"""
    try:
        folder_path = Path(path).expanduser().absolute()
        
        if not folder_path.exists():
            return {"error": "Path does not exist", "items": []}
        
        items = []
        for item in sorted(folder_path.iterdir()):
            try:
                if item.is_dir() and not item.name.startswith('.'):
                    # Check if it looks like a dataset
                    is_dataset = False
                    dataset_format = None
                    
                    # Quick detection
                    if list(item.glob("*.yaml")) or list(item.glob("labels/*.txt")):
                        is_dataset = True
                        dataset_format = "yolo"
                    elif list(item.glob("*.json")):
                        is_dataset = True
                        dataset_format = "coco/labelme"
                    elif list(item.glob("*.xml")):
                        is_dataset = True
                        dataset_format = "pascal-voc"
                    
                    items.append({
                        "name": item.name,
                        "path": str(item),
                        "is_dataset": is_dataset,
                        "format_hint": dataset_format,
                        "type": "directory"
                    })
            except PermissionError:
                continue
        
        return {
            "current_path": str(folder_path),
            "parent_path": str(folder_path.parent),
            "items": items
        }
    
    except Exception as e:
        return {"error": str(e), "items": []}


# ============== VIDEO FRAME EXTRACTION ==============

VIDEOS_DIR = WORKSPACE_DIR / "videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

# Store uploaded video info
_uploaded_videos: Dict[str, Dict[str, Any]] = {}


class VideoExtractRequest(BaseModel):
    video_id: Optional[str] = None  # For uploaded videos
    video_path: Optional[str] = None  # For local path
    output_name: str
    mode: str = "interval"  # interval, uniform, keyframes, manual
    nth_frame: int = 30
    frame_interval: Optional[int] = None
    uniform_count: Optional[int] = 100
    manual_frames: Optional[List[int]] = None
    max_frames: Optional[int] = None
    start_time: float = 0
    end_time: Optional[float] = None


@app.post("/api/videos/upload")
async def upload_video(video: UploadFile = File(...)):
    """Upload a video file for frame extraction"""
    video_id = str(uuid.uuid4())[:8]
    video_filename = video.filename or f"video_{video_id}.mp4"
    video_path = VIDEOS_DIR / f"{video_id}_{video_filename}"
    
    # Save the uploaded video
    content = await video.read()
    with open(video_path, "wb") as f:
        f.write(content)
    
    # Get video info
    info = video_extractor.get_video_info(video_path)
    
    video_data = {
        "id": video_id,
        "filename": video_filename,
        "path": str(video_path),
        "url": f"/api/videos/{video_id}/stream",
        "duration": info.get("duration", 0) if info.get("success") else 60,
        "fps": info.get("fps", 30) if info.get("success") else 30,
        "total_frames": info.get("total_frames", 1800) if info.get("success") else 1800,
        "width": info.get("width", 1920) if info.get("success") else 1920,
        "height": info.get("height", 1080) if info.get("success") else 1080,
        "thumbnail": None
    }
    
    _uploaded_videos[video_id] = video_data
    
    return video_data


@app.get("/api/videos/{video_id}/stream")
async def stream_video(video_id: str):
    """Stream a video file"""
    if video_id not in _uploaded_videos:
        raise HTTPException(status_code=404, detail="Video not found")
    
    video_path = Path(_uploaded_videos[video_id]["path"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(video_path, media_type="video/mp4")


@app.post("/api/video/extract")
@app.post("/api/videos/extract-frames")
async def extract_video_frames(request: VideoExtractRequest):
    """Extract frames from a video file to create a dataset"""
    # Determine video path
    if request.video_id and request.video_id in _uploaded_videos:
        video_path = Path(_uploaded_videos[request.video_id]["path"])
    elif request.video_path:
        video_path = Path(request.video_path)
    else:
        raise HTTPException(status_code=400, detail="No video specified")
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    # Calculate nth_frame based on mode
    nth_frame = request.nth_frame
    if request.frame_interval:
        nth_frame = request.frame_interval
    
    # Create new dataset
    dataset_id = str(uuid.uuid4())
    output_path = DATASETS_DIR / dataset_id / "images"
    output_path.mkdir(parents=True, exist_ok=True)
    
    result = video_extractor.extract_frames(
        video_path,
        output_path,
        nth_frame=nth_frame,
        max_frames=request.max_frames or request.uniform_count,
        start_time=request.start_time,
        end_time=request.end_time
    )
    
    if not result["success"]:
        shutil.rmtree(DATASETS_DIR / dataset_id, ignore_errors=True)
        raise HTTPException(status_code=400, detail=result.get("error", "Extraction failed"))
    
    # Create YOLO-style dataset structure
    dataset_path = DATASETS_DIR / dataset_id
    (dataset_path / "labels").mkdir(exist_ok=True)
    
    # Create data.yaml
    config = {
        "path": str(dataset_path.absolute()),
        "train": "images",
        "val": "images",
        "names": {},
        "nc": 0
    }
    with open(dataset_path / "data.yaml", "w") as f:
        yaml.dump(config, f)
    
    # Parse as dataset
    dataset_info = dataset_parser.parse_dataset(dataset_path, "yolo", request.output_name)
    dataset_info["id"] = dataset_id
    dataset_info["source_video"] = str(video_path)
    active_datasets[dataset_id] = dataset_info
    _save_dataset_metadata(dataset_id, dataset_info)
    
    return {
        "success": True,
        "new_dataset": dataset_info,
        "dataset": dataset_info,
        "dataset_name": request.output_name,
        "extracted_frames": result["extracted_frames"],
        "video_info": result.get("video_info", {})
    }


@app.get("/api/video/info")
async def get_video_info(video_path: str):
    """Get information about a video file"""
    result = video_extractor.get_video_info(Path(video_path))
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to get video info"))
    return result


# ============== DUPLICATE DETECTION ==============

class DuplicateDetectionRequest(BaseModel):
    method: str = "perceptual"  # "md5", "perceptual", "average", "clip"
    threshold: int = 10
    include_near_duplicates: bool = True


@app.post("/api/datasets/{dataset_id}/find-duplicates")
async def find_duplicate_images(dataset_id: str, request: DuplicateDetectionRequest):
    """Find duplicate or similar images in a dataset"""
    if dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset_path = DATASETS_DIR / dataset_id
    
    if request.method == "clip":
        result = clip_manager.find_similar_images(
            dataset_path,
            similarity_threshold=request.threshold / 100.0
        )
        # Normalize CLIP result keys to match hash-based result structure
        if result.get("success"):
            result["duplicate_groups"] = result.pop("similar_groups", 0)
            result["total_duplicates"] = result.pop("total_similar", 0)
            result["unique_images"] = result.get("total_images", 0) - result["total_duplicates"]
            result["method"] = "clip"
            result["threshold"] = request.threshold
    else:
        result = duplicate_detector.find_duplicates(
            dataset_path,
            method=request.method,
            threshold=request.threshold,
            include_near_duplicates=request.include_near_duplicates
        )

    return result


class RemoveDuplicatesRequest(BaseModel):
    groups: List[List[Dict[str, Any]]]
    keep_strategy: str = "first"  # "first", "largest", "smallest"


@app.post("/api/datasets/{dataset_id}/remove-duplicates")
async def remove_duplicate_images(dataset_id: str, request: RemoveDuplicatesRequest):
    """Remove duplicate images from a dataset"""
    if dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = active_datasets[dataset_id]
    dataset_path = DATASETS_DIR / dataset_id
    
    result = duplicate_detector.remove_duplicates(
        dataset_path,
        dataset["format"],
        request.groups,
        request.keep_strategy
    )
    
    # Update dataset info
    dataset_info = dataset_parser.parse_dataset(dataset_path, dataset["format"], dataset["name"])
    dataset_info["id"] = dataset_id
    active_datasets[dataset_id] = dataset_info
    
    return {**result, "updated_dataset": dataset_info}


# ============== ENHANCED AUGMENTATION ==============

@app.get("/api/augmentations/list")
async def list_augmentation_options():
    """Get all available augmentation options with their parameters"""
    return {"augmentations": augmenter.get_available_augmentations()}


class EnhancedAugmentationRequest(BaseModel):
    output_name: str
    target_size: int
    target_multiplier: Optional[float] = None  # Alternative: multiply dataset by X
    augmentations: Dict[str, Dict[str, Any]]
    preserve_originals: bool = True


@app.post("/api/datasets/{dataset_id}/augment-enhanced")
async def augment_dataset_enhanced(dataset_id: str, request: EnhancedAugmentationRequest):
    """Augment dataset with comprehensive options"""
    if dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = active_datasets[dataset_id]
    dataset_path = DATASETS_DIR / dataset_id
    
    # Calculate target size
    target_size = request.target_size
    if request.target_multiplier:
        target_size = int(dataset["num_images"] * request.target_multiplier)
    
    # Create new augmented dataset
    new_dataset_id = str(uuid.uuid4())
    output_path = DATASETS_DIR / new_dataset_id
    
    result = augmenter.augment_dataset(
        dataset_path,
        output_path,
        dataset["format"],
        target_size,
        request.augmentations,
        request.preserve_originals
    )
    
    if not result["success"]:
        shutil.rmtree(output_path, ignore_errors=True)
        raise HTTPException(status_code=400, detail=result.get("error", "Augmentation failed"))
    
    # Parse new dataset
    new_info = dataset_parser.parse_dataset(output_path, dataset["format"], request.output_name)
    new_info["id"] = new_dataset_id
    active_datasets[new_dataset_id] = new_info
    
    return {
        "success": True,
        "new_dataset": new_info,
        "original_images": result["original_images"],
        "augmented_images": result["augmented_images"],
        "total_images": result["total_images"]
    }


# ============== ENHANCED SPLITTING ==============

class EnhancedSplitRequest(BaseModel):
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    output_name: Optional[str] = None
    shuffle: bool = True
    seed: Optional[int] = None
    stratified: bool = False  # Maintain class distribution in splits
    by_folder: bool = False  # Split by existing folders


@app.post("/api/datasets/{dataset_id}/split-enhanced")
async def split_dataset_enhanced(dataset_id: str, request: EnhancedSplitRequest):
    """Split dataset with enhanced options including stratified splitting"""
    if dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Validate ratios
    total_ratio = request.train_ratio + request.val_ratio + request.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise HTTPException(status_code=400, detail="Ratios must sum to 1.0")
    
    dataset = active_datasets[dataset_id]
    dataset_path = DATASETS_DIR / dataset_id
    
    # Create new dataset with splits
    new_dataset_id = str(uuid.uuid4())
    output_path = DATASETS_DIR / new_dataset_id
    
    split_result = dataset_parser.create_split_dataset(
        dataset_path,
        output_path,
        dataset["format"],
        train_ratio=request.train_ratio,
        val_ratio=request.val_ratio,
        test_ratio=request.test_ratio,
        shuffle=request.shuffle,
        seed=request.seed,
        stratified=request.stratified
    )
    
    # Parse new dataset
    output_name = request.output_name or f"{dataset['name']}_split"
    new_info = dataset_parser.parse_dataset(output_path, dataset["format"], output_name)
    new_info["id"] = new_dataset_id
    new_info["splits"] = split_result["splits"]
    active_datasets[new_dataset_id] = new_info
    
    return {
        "success": True,
        "new_dataset": new_info,
        "splits": split_result["splits"],
        "class_distribution": split_result.get("class_distribution", {})
    }


# ============== FORMAT INFORMATION ==============

@app.get("/api/formats")
async def list_formats():
    """List all supported annotation formats"""
    return {
        "formats": [
            {"id": "yolo", "name": "YOLO", "extensions": [".txt"], "task": ["detection", "segmentation"]},
            {"id": "yolov5", "name": "YOLOv5", "extensions": [".txt"], "task": ["detection", "segmentation"]},
            {"id": "yolov8", "name": "YOLOv8", "extensions": [".txt"], "task": ["detection", "segmentation", "pose"]},
            {"id": "yolov9", "name": "YOLOv9", "extensions": [".txt"], "task": ["detection", "segmentation"]},
            {"id": "yolov10", "name": "YOLOv10", "extensions": [".txt"], "task": ["detection"]},
            {"id": "yolov11", "name": "YOLOv11", "extensions": [".txt"], "task": ["detection", "segmentation"]},
            {"id": "coco", "name": "COCO JSON", "extensions": [".json"], "task": ["detection", "segmentation", "keypoint"]},
            {"id": "pascal-voc", "name": "Pascal VOC", "extensions": [".xml"], "task": ["detection"]},
            {"id": "labelme", "name": "LabelMe", "extensions": [".json"], "task": ["segmentation", "detection"]},
            {"id": "createml", "name": "CreateML", "extensions": [".json"], "task": ["detection", "classification"]},
            {"id": "tensorflow-csv", "name": "TensorFlow CSV", "extensions": [".csv"], "task": ["detection"]},
            {"id": "classification-folder", "name": "Classification Folder", "extensions": [], "task": ["classification"]},
            {"id": "supervisely", "name": "Supervisely", "extensions": [".json"], "task": ["detection", "segmentation"]},
            {"id": "cvat", "name": "CVAT", "extensions": [".xml"], "task": ["detection", "segmentation"]},
        ]
    }


# ============== LOCAL FOLDER BROWSING ==============

class LocalFolderRequest(BaseModel):
    path: str
    dataset_name: Optional[str] = None
    format_hint: Optional[str] = None


@app.post("/api/browse-folders")
async def browse_local_folders(path: str = "."):
    """Browse local filesystem folders"""
    try:
        folder_path = Path(path).expanduser().resolve()
        
        if not folder_path.exists():
            return {"error": "Path does not exist", "items": [], "current_path": str(folder_path)}
        
        items = []
        
        # Add parent directory option
        if folder_path.parent != folder_path:
            items.append({
                "name": "..",
                "path": str(folder_path.parent),
                "is_directory": True,
                "is_dataset": False
            })
        
        for item in sorted(folder_path.iterdir()):
            try:
                if item.name.startswith('.'):
                    continue
                    
                if item.is_dir():
                    # Check if it looks like a dataset
                    is_dataset = False
                    format_hint = None
                    
                    # Quick detection
                    if list(item.glob("*.yaml")) or list(item.glob("data.yaml")):
                        is_dataset = True
                        format_hint = "yolo"
                    elif list(item.glob("*.json")):
                        is_dataset = True
                        format_hint = "coco/labelme"
                    elif list(item.glob("*.xml")):
                        is_dataset = True
                        format_hint = "pascal-voc"
                    elif any(d.is_dir() and list(d.glob("*.jpg")) or list(d.glob("*.png")) for d in item.iterdir() if d.is_dir()):
                        is_dataset = True
                        format_hint = "classification-folder"
                    
                    items.append({
                        "name": item.name,
                        "path": str(item),
                        "is_directory": True,
                        "is_dataset": is_dataset,
                        "format_hint": format_hint
                    })
            except PermissionError:
                continue
        
        return {
            "current_path": str(folder_path),
            "parent_path": str(folder_path.parent) if folder_path.parent != folder_path else None,
            "items": items
        }
    
    except Exception as e:
        return {"error": str(e), "items": [], "current_path": path}


@app.post("/api/datasets/load-local")
async def load_local_dataset(request: LocalFolderRequest):
    """Load a dataset directly from a local folder path (no upload required)"""
    folder_path = Path(request.path).expanduser().resolve()
    
    if not folder_path.exists():
        raise HTTPException(status_code=404, detail=f"Folder not found: {request.path}")
    
    if not folder_path.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")
    
    dataset_id = str(uuid.uuid4())
    
    # Create symlink to avoid copying large datasets
    dataset_link = DATASETS_DIR / dataset_id
    
    try:
        # Try symlink first (faster, saves space)
        try:
            dataset_link.symlink_to(folder_path)
        except OSError:
            # If symlink fails (e.g., Windows without admin), copy the dataset
            shutil.copytree(folder_path, dataset_link)
        
        # Parse dataset
        dataset_info = dataset_parser.parse_dataset(
            dataset_link,
            format_hint=request.format_hint,
            name=request.dataset_name or folder_path.name
        )
        
        dataset_info["id"] = dataset_id
        dataset_info["local_path"] = str(folder_path)
        dataset_info["is_local"] = True
        active_datasets[dataset_id] = dataset_info
        
        return {"success": True, "dataset": dataset_info}
    
    except Exception as e:
        # Cleanup on failure
        if dataset_link.exists():
            if dataset_link.is_symlink():
                dataset_link.unlink()
            else:
                shutil.rmtree(dataset_link, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(e))


# ============== IMAGE SERVING ==============

@app.get("/api/datasets/{dataset_id}/image-file/{image_path:path}")
async def serve_dataset_image(dataset_id: str, image_path: str):
    """Serve an image file from a dataset"""
    if dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Resolve nested subdirectory (e.g. when ZIP extracted into a single folder)
    dataset_root = dataset_parser._find_dataset_root(DATASETS_DIR / dataset_id)
    image_file = dataset_root / image_path

    if not image_file.exists():
        # Fallback: try relative to raw dataset dir
        image_file = DATASETS_DIR / dataset_id / image_path
    if not image_file.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    # Always include CORS header so crossOrigin='anonymous' canvas loads work
    # even when the same URL was previously cached by a plain <img> request.
    return FileResponse(image_file, headers={"Access-Control-Allow-Origin": "*"})


# Health check

# ── App Settings ───────────────────────────────────────────────────────────────
# In-memory store (persists for the lifetime of the server process).
# The frontend also keeps a localStorage copy so nothing is lost on restart.
_app_settings: dict = {
    "models_path": str(MODELS_DIR),
    "datasets_path": str(DATASETS_DIR),
    "output_path": str(EXPORTS_DIR),
    "use_gpu": True,
    "gpu_device": "0",
}


@app.get("/api/settings")
async def get_settings():
    """Return current runtime settings."""
    return _app_settings


@app.post("/api/settings")
async def update_settings(config: SettingsConfig):
    """
    Update runtime paths and hardware settings.
    Path changes are validated and applied to the global directory variables
    so that subsequent operations use the new locations.
    """
    global DATASETS_DIR, MODELS_DIR, EXPORTS_DIR, _app_settings

    if config.datasets_path:
        new_path = Path(config.datasets_path).expanduser()
        new_path.mkdir(parents=True, exist_ok=True)
        DATASETS_DIR = new_path
        _app_settings["datasets_path"] = str(new_path)

    if config.models_path:
        new_path = Path(config.models_path).expanduser()
        new_path.mkdir(parents=True, exist_ok=True)
        MODELS_DIR = new_path
        _app_settings["models_path"] = str(new_path)

    if config.output_path:
        new_path = Path(config.output_path).expanduser()
        new_path.mkdir(parents=True, exist_ok=True)
        EXPORTS_DIR = new_path
        _app_settings["output_path"] = str(new_path)

    _app_settings["use_gpu"] = config.use_gpu
    _app_settings["gpu_device"] = config.gpu_device

    return {"success": True, "settings": _app_settings}



@app.post("/api/restart")
async def restart_server():
    """
    Gracefully restart the backend process.
    Because uvicorn --reload is running, we simply touch main.py —
    uvicorn's file-watcher detects the change and reloads automatically
    without dropping the port.  This is zero-downtime for Python changes.
    """
    import threading
    def _touch():
        import time, pathlib
        time.sleep(0.3)
        p = pathlib.Path(__file__)
        p.touch()   # update mtime → triggers uvicorn --reload
    threading.Thread(target=_touch, daemon=True).start()
    return {"success": True, "message": "Reload triggered — uvicorn will restart in ~1 s"}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "version": "3.0.0"}


@app.get("/api/device-info")
async def device_info():
    """Return compute device info and CUDA install status."""
    info: dict = {
        "device": model_manager._get_device(),
        "cuda_available": False,
        "cuda_version": None,
        "gpus": [],
        "gpu_status": _gpu_status,
    }
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        info["torch_version"] = "not installed"
    return info


@app.get("/api/system")
async def system_stats():
    """Return real CPU, RAM, and GPU usage stats."""
    import importlib

    stats: dict = {
        "cpu_percent": 0.0,
        "ram_percent": 0.0,
        "gpu_percent": None,
        "gpu_memory_percent": None,
        "gpu_name": None,
    }

    # --- CPU / RAM via psutil (optional dependency) ---
    psutil_spec = importlib.util.find_spec("psutil")
    if psutil_spec is not None:
        import psutil  # type: ignore
        stats["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        vm = psutil.virtual_memory()
        stats["ram_percent"] = vm.percent

    # --- GPU via pynvml (optional, NVIDIA only) ---
    pynvml_spec = importlib.util.find_spec("pynvml")
    if pynvml_spec is not None:
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            stats["gpu_percent"] = float(util.gpu)
            stats["gpu_memory_percent"] = round(mem_info.used / mem_info.total * 100, 1)
            stats["gpu_name"] = name if isinstance(name, str) else name.decode()
        except Exception:
            pass  # No NVIDIA GPU or driver not available

    return stats


@app.get("/api/hardware")
async def hardware_info():
    """Return static hardware capabilities: CUDA, GPU specs, CPU, RAM."""
    import importlib, platform

    info: dict = {
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "cuda_available": False,
        "cuda_version": None,
        "gpu_count": 0,
        "gpus": [],          # [{name, memory_total_mb, memory_free_mb}]
        "cpu_cores": None,
        "cpu_model": None,
        "ram_total_gb": None,
        "ram_available_gb": None,
    }

    # CPU / RAM via psutil
    psutil_spec = importlib.util.find_spec("psutil")
    if psutil_spec is not None:
        import psutil  # type: ignore
        info["cpu_cores"] = psutil.cpu_count(logical=True)
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / 1024 ** 3, 1)
        info["ram_available_gb"] = round(vm.available / 1024 ** 3, 1)

    # CPU model name (cross-platform)
    try:
        if platform.system() == "Windows":
            import subprocess
            r = subprocess.run(
                ["wmic", "cpu", "get", "Name", "/value"],
                capture_output=True, text=True, timeout=3
            )
            for line in r.stdout.splitlines():
                if line.startswith("Name="):
                    info["cpu_model"] = line.split("=", 1)[1].strip()
                    break
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        info["cpu_model"] = line.split(":", 1)[1].strip()
                        break
        elif platform.system() == "Darwin":
            import subprocess
            r = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=3
            )
            info["cpu_model"] = r.stdout.strip()
    except Exception:
        pass

    # CUDA / GPU via torch
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is not None:
        try:
            import torch  # type: ignore
            info["cuda_available"] = torch.cuda.is_available()
            if info["cuda_available"]:
                info["cuda_version"] = torch.version.cuda
                info["gpu_count"] = torch.cuda.device_count()
                for i in range(info["gpu_count"]):
                    props = torch.cuda.get_device_properties(i)
                    mem_total = props.total_memory // (1024 ** 2)
                    try:
                        free, _ = torch.cuda.mem_get_info(i)
                        mem_free = free // (1024 ** 2)
                    except Exception:
                        mem_free = None
                    info["gpus"].append({
                        "index": i,
                        "name": props.name,
                        "memory_total_mb": mem_total,
                        "memory_free_mb": mem_free,
                    })
        except Exception:
            pass

    # Fallback: pynvml
    if not info["gpus"] and importlib.util.find_spec("pynvml") is not None:
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            info["gpu_count"] = count
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                info["gpus"].append({
                    "index": i,
                    "name": name if isinstance(name, str) else name.decode(),
                    "memory_total_mb": mem.total // (1024 ** 2),
                    "memory_free_mb": mem.free // (1024 ** 2),
                })
            if info["gpus"]:
                info["cuda_available"] = True  # NVIDIA GPU present, even if torch CUDA missing
        except Exception:
            pass

    # Final fallback: nvidia-smi (works even without Python GPU packages)
    if not info["gpus"]:
        try:
            import subprocess
            r = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,name,memory.total,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0:
                for line in r.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        info["gpus"].append({
                            "index": int(parts[0]),
                            "name": parts[1],
                            "memory_total_mb": int(parts[2]),
                            "memory_free_mb": int(parts[3]),
                        })
                if info["gpus"]:
                    info["gpu_count"] = len(info["gpus"])
                    info["cuda_available"] = True
                    # Try to get CUDA version from nvidia-smi header
                    hdr = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
                    for line in hdr.stdout.splitlines():
                        if "CUDA Version:" in line:
                            import re
                            m = re.search(r"CUDA Version:\s*([\d.]+)", line)
                            if m:
                                info["cuda_version"] = m.group(1)
                            break
        except Exception:
            pass

    return info


class YamlWizardConfig(BaseModel):
    class_names: List[str]          # ordered list, index = class_id
    train_path: Optional[str] = None
    val_path:   Optional[str] = None
    test_path:  Optional[str] = None


@app.get("/api/datasets/{dataset_id}/class-samples")
async def get_class_samples(dataset_id: str, samples_per_class: int = 3):
    """
    Return sample images for each unique class_id found in the dataset,
    plus detected split directory paths.  Powers the YAML Wizard.
    """
    if dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset   = active_datasets[dataset_id]
    raw_path  = DATASETS_DIR / dataset_id
    root      = dataset_parser._find_dataset_root(raw_path)
    fmt       = dataset["format"]

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    # ── Collect per-class samples from YOLO label files ─────────────────────
    class_samples: Dict[int, list] = {}   # class_id → [{image_path, annotations}]

    if fmt in ("yolo", "yolov5", "yolov8", "yolov9", "yolov10", "yolov11", "yolov12"):
        # Load class names from yaml
        existing_names: List[str] = []
        for yf in list(root.glob("*.yaml")) + list(root.glob("*.yml")):
            try:
                import yaml as _yaml
                with open(yf) as f:
                    cfg = _yaml.safe_load(f)
                if cfg and "names" in cfg:
                    n = cfg["names"]
                    existing_names = list(n.values()) if isinstance(n, dict) else list(n)
                break
            except Exception:
                pass

        # Walk label dirs
        for label_file in root.glob("**/labels/*.txt"):
            img_file = None
            img_stem  = label_file.stem
            img_dir   = label_file.parent.parent / "images"
            for ext in IMAGE_EXTS:
                cand = img_dir / (img_stem + ext)
                if cand.exists():
                    img_file = cand
                    break
            if img_file is None:
                continue

            try:
                annotations = []
                with open(label_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cid = int(parts[0])
                        if len(parts) == 5:
                            annotations.append({
                                "type": "bbox", "class_id": cid,
                                "x_center": float(parts[1]), "y_center": float(parts[2]),
                                "width": float(parts[3]), "height": float(parts[4]),
                                "normalized": True,
                            })
                        else:
                            annotations.append({
                                "type": "polygon", "class_id": cid,
                                "points": [float(p) for p in parts[1:]],
                                "normalized": True,
                            })
                if not annotations:
                    continue
                # Register each class that appears in this file
                seen_cids = set(a["class_id"] for a in annotations)
                rel_path = str(img_file.relative_to(raw_path))
                for cid in seen_cids:
                    if cid not in class_samples:
                        class_samples[cid] = []
                    if len(class_samples[cid]) < samples_per_class:
                        class_samples[cid].append({
                            "image_id":   img_stem,
                            "image_path": rel_path,
                            "annotations": annotations,
                        })
            except Exception:
                continue

    # ── Detect split paths ──────────────────────────────────────────────────
    def _rel(p: Path) -> str:
        return "../" + str(p.relative_to(root.parent)).replace("\\", "/")

    splits: Dict[str, Optional[str]] = {"train": None, "val": None, "test": None}
    for split, aliases in [("train", ["train"]), ("val", ["val", "valid", "validation"]), ("test", ["test"])]:
        for alias in aliases:
            for sub in ["images", ""]:
                cand = root / alias / sub if sub else root / alias
                if cand.exists() and cand.is_dir():
                    splits[split] = _rel(cand)
                    break
            if splits[split]:
                break

    return {
        "classes": [
            {
                "class_id": cid,
                "existing_name": existing_names[cid] if cid < len(existing_names) else None,
                "samples": class_samples.get(cid, []),
            }
            for cid in sorted(class_samples.keys())
        ],
        "splits": splits,
        "dataset_path": str(root),
        "existing_names": existing_names,
    }


@app.post("/api/datasets/{dataset_id}/generate-yaml")
async def generate_dataset_yaml(dataset_id: str, config: YamlWizardConfig):
    """Write data.yaml with the provided class names and split paths."""
    if dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    raw_path = DATASETS_DIR / dataset_id
    root     = dataset_parser._find_dataset_root(raw_path)

    import yaml as _yaml

    yaml_data: dict = {}
    yaml_data["path"] = str(root)
    if config.train_path:
        yaml_data["train"] = config.train_path
    if config.val_path:
        yaml_data["val"] = config.val_path
    if config.test_path:
        yaml_data["test"] = config.test_path

    yaml_data["nc"]    = len(config.class_names)
    yaml_data["names"] = {i: name for i, name in enumerate(config.class_names)}

    out_file = root / "data.yaml"
    with open(out_file, "w") as f:
        _yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

    # Render preview string
    lines = [f"path: {yaml_data['path']}"]
    for k in ("train", "val", "test"):
        if k in yaml_data:
            lines.append(f"{k}: {yaml_data[k]}")
    lines += ["", f"nc: {yaml_data['nc']}", f"names: {config.class_names}"]
    preview = "\n".join(lines)

    # Refresh dataset metadata
    dinfo = dataset_parser.parse_dataset(raw_path, active_datasets[dataset_id]["format"],
                                         active_datasets[dataset_id]["name"])
    dinfo["id"] = dataset_id
    active_datasets[dataset_id] = dinfo

    return {"success": True, "yaml_path": str(out_file), "preview": preview}


class BatchDeleteRequest(BaseModel):
    image_ids: List[str]

class BatchSplitRequest(BaseModel):
    image_ids: List[str]
    split: str

class SnapshotRequest(BaseModel):
    name: str
    description: str = ""


@app.post("/api/datasets/{dataset_id}/images/batch-delete")
async def batch_delete_images(dataset_id: str, request: BatchDeleteRequest):
    """Delete a batch of images (and their annotations) from a dataset by image ID."""
    if dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = active_datasets[dataset_id]
    dataset_path = DATASETS_DIR / dataset_id
    fmt = dataset["format"]
    ids_to_delete = set(request.image_ids)

    # Build full image list from cache or parser
    all_images = _images_cache.get(dataset_id) or dataset_parser.get_images_with_annotations(
        dataset_path, fmt, page=1, limit=999999
    )

    deleted = 0
    for img in all_images:
        if img["id"] not in ids_to_delete:
            continue
        img_path = dataset_path / img["path"]
        # Delete image file
        try:
            if img_path.exists():
                img_path.unlink()
        except Exception:
            pass
        # Delete annotation file (YOLO .txt sidecar)
        if fmt in ("yolo", "yolov5", "yolov8", "yolov9", "yolov10", "yolov11"):
            label_path = img_path.with_suffix(".txt")
            # Labels are typically in labels/ not images/
            label_path2 = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
            for lp in (label_path, label_path2):
                try:
                    if lp.exists():
                        lp.unlink()
                except Exception:
                    pass
        # Pascal VOC: delete xml in Annotations/
        elif fmt in ("pascal-voc", "voc"):
            ann_path = dataset_path / "Annotations" / (img_path.stem + ".xml")
            try:
                if ann_path.exists():
                    ann_path.unlink()
            except Exception:
                pass
        deleted += 1

    # Bust cache and refresh dataset stats
    _images_cache.pop(dataset_id, None)
    try:
        dataset_info = dataset_parser.parse_dataset(dataset_path, fmt, dataset["name"])
        dataset_info["id"] = dataset_id
        active_datasets[dataset_id] = dataset_info
    except Exception:
        pass

    return {"success": True, "deleted": deleted}


@app.post("/api/datasets/{dataset_id}/images/batch-split")
async def batch_assign_split(dataset_id: str, request: BatchSplitRequest):
    """Move a batch of images to a different split (train/val/test)."""
    if dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = active_datasets[dataset_id]
    dataset_path = DATASETS_DIR / dataset_id
    fmt = dataset["format"]
    target_split = request.split.strip().lower()
    ids_to_move = set(request.image_ids)

    if not target_split:
        raise HTTPException(status_code=400, detail="split must not be empty")

    all_images = _images_cache.get(dataset_id) or dataset_parser.get_images_with_annotations(
        dataset_path, fmt, page=1, limit=999999
    )

    moved = 0
    for img in all_images:
        if img["id"] not in ids_to_move:
            continue
        current_path = dataset_path / img["path"]
        if not current_path.exists():
            continue

        # Determine new image path (swap split segment)
        rel = Path(img["path"])
        parts = rel.parts
        if len(parts) >= 2 and parts[0] in ("images", "train", "val", "valid", "test"):
            # e.g. images/train/foo.jpg  or  train/foo.jpg
            if parts[0] == "images" and len(parts) >= 3:
                new_rel = Path("images") / target_split / Path(*parts[2:])
                label_old = Path("labels") / parts[1] / Path(*parts[2:])
                label_new = Path("labels") / target_split / Path(*parts[2:])
            else:
                new_rel = Path(target_split) / Path(*parts[1:])
                label_old = None
                label_new = None
        else:
            continue  # Can't determine split structure

        new_path = dataset_path / new_rel
        new_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import shutil as _shutil
            _shutil.move(str(current_path), str(new_path))
        except Exception:
            continue

        # Move annotation file for YOLO
        if fmt in ("yolo", "yolov5", "yolov8", "yolov9", "yolov10", "yolov11") and label_old and label_new:
            old_label_path = dataset_path / label_old.with_suffix(".txt")
            new_label_path = dataset_path / label_new.with_suffix(".txt")
            if old_label_path.exists():
                new_label_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    import shutil as _shutil
                    _shutil.move(str(old_label_path), str(new_label_path))
                except Exception:
                    pass

        moved += 1

    # Bust cache and refresh
    _images_cache.pop(dataset_id, None)
    try:
        dataset_info = dataset_parser.parse_dataset(dataset_path, fmt, dataset["name"])
        dataset_info["id"] = dataset_id
        active_datasets[dataset_id] = dataset_info
    except Exception:
        pass

    return {"success": True, "moved": moved, "split": target_split}


# ============== SNAPSHOTS ==============

@app.post("/api/datasets/{dataset_id}/snapshot")
async def create_snapshot(dataset_id: str, request: SnapshotRequest):
    """Create a snapshot (zip export) of a dataset at current state."""
    if dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    import zipfile, uuid as _uuid, time as _time

    dataset = active_datasets[dataset_id]
    dataset_path = DATASETS_DIR / dataset_id
    snapshot_id = f"snap_{int(_time.time() * 1000)}"
    snap_dir = SNAPSHOTS_DIR / dataset_id
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_zip = snap_dir / f"{snapshot_id}.zip"

    try:
        with zipfile.ZipFile(snap_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in dataset_path.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(dataset_path))
        size_mb = snap_zip.stat().st_size / (1024 * 1024)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create snapshot: {e}")

    return {
        "success": True,
        "snapshot_id": snapshot_id,
        "size_mb": round(size_mb, 2),
    }


@app.get("/api/datasets/{dataset_id}/snapshot/{snapshot_id}/download")
async def download_snapshot(dataset_id: str, snapshot_id: str):
    """Download a snapshot zip file."""
    snap_zip = SNAPSHOTS_DIR / dataset_id / f"{snapshot_id}.zip"
    if not snap_zip.exists():
        # Fall back to the standard export
        raise HTTPException(status_code=404, detail="Snapshot not found")
    from fastapi.responses import FileResponse
    return FileResponse(
        path=str(snap_zip),
        media_type="application/zip",
        filename=f"{snapshot_id}.zip",
    )


@app.post("/api/datasets/{dataset_id}/snapshot/{snapshot_id}/restore")
async def restore_snapshot(dataset_id: str, snapshot_id: str):
    """Restore a dataset from a snapshot zip."""
    if dataset_id not in active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    snap_zip = SNAPSHOTS_DIR / dataset_id / f"{snapshot_id}.zip"
    if not snap_zip.exists():
        raise HTTPException(status_code=404, detail="Snapshot not found")

    import zipfile, shutil as _shutil

    dataset_path = DATASETS_DIR / dataset_id
    dataset = active_datasets[dataset_id]
    fmt = dataset["format"]
    name = dataset["name"]

    # Wipe current dataset files and re-extract snapshot
    try:
        _shutil.rmtree(dataset_path)
        dataset_path.mkdir(parents=True)
        with zipfile.ZipFile(snap_zip, "r") as zf:
            zf.extractall(dataset_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Restore failed: {e}")

    # Bust cache and re-parse
    _images_cache.pop(dataset_id, None)
    try:
        dataset_info = dataset_parser.parse_dataset(dataset_path, fmt, name)
        dataset_info["id"] = dataset_id
        active_datasets[dataset_id] = dataset_info
    except Exception:
        pass

    return {"success": True}


@app.post("/api/shutdown")
async def shutdown_server():
    """Gracefully shut down the backend server"""
    import threading, os, signal
    def _kill():
        import time
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=_kill, daemon=True).start()
    return {"success": True, "message": "Server shutting down"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
