# VisOS — Complete Documentation

## Table of Contents

1. [Getting Started](#getting-started)
2. [Dataset Management](#dataset-management)
3. [Sorting & Filtering](#sorting--filtering)
4. [Annotation Tools](#annotation-tools)
5. [Class Management](#class-management)
6. [Data Augmentation](#data-augmentation)
7. [Video Frame Extraction](#video-frame-extraction)
8. [Duplicate Detection](#duplicate-detection)
9. [Dataset Splitting](#dataset-splitting)
10. [Format Conversion](#format-conversion)
11. [Dataset Merging](#dataset-merging)
12. [Model Management](#model-management)
13. [Batch Jobs](#batch-jobs)
14. [Settings](#settings)
15. [API Reference](#api-reference)

---

## Getting Started

### System Requirements

**Minimum:**
- CPU: Intel i5 / AMD Ryzen 5 or equivalent
- RAM: 8 GB
- Storage: 10 GB free space
- OS: Windows 10+, macOS 11+, Ubuntu 20.04+

**Recommended (for SAM 3 inference):**
- CPU: Intel i7 / AMD Ryzen 7 or better
- RAM: 16 GB+
- GPU: NVIDIA RTX 3060+ with 8 GB+ VRAM
- Storage: SSD with 10 GB+ free space (SAM 3 weights are ~3.5 GB)

### First Launch

1. Start the application:
   ```bash
   uv run app.py
   ```

2. Open `http://localhost:3000` in your browser.

3. **First step**: Go to **Settings** and verify the backend connection shows "Connected".

4. **Load your first dataset**: Click **Datasets** → **Open Local Folder** → navigate to your dataset directory.

### Process Manager

`app.py` starts both the FastAPI backend and the Next.js frontend. Ctrl-C stops them. There is no separate `restart` command — stop and re-run.

### Understanding the Interface

- **Sidebar (left)**: Navigation grouped into Data, Annotate, Process, and Analyze sections
- **Main area (center)**: The active view
- **Dataset indicator**: Shows the currently selected dataset

---

## Dataset Management

### Loading a Dataset

**From local folder (recommended):**
1. Go to **Datasets**
2. Click **Open Local Folder**
3. Browse to your dataset directory
4. Format is auto-detected — no configuration needed
5. Click **Load**

**By uploading a ZIP:**
1. Go to **Datasets**
2. Click **Upload Dataset**
3. Select a ZIP file containing your dataset
4. Wait for upload and processing

### Supported Formats

VisOS auto-detects all of the following on load, and can export to all of them:

| Format | Notes |
|---|---|
| YOLO | Requires `data.yaml`, `images/`, `labels/` |
| YOLO OBB | Oriented bounding boxes |
| COCO | `annotations/*.json` |
| COCO Panoptic | Panoptic segmentation variant |
| Pascal VOC | `Annotations/*.xml`, `JPEGImages/` |
| LabelMe | Per-image JSON files |
| CreateML | Apple Create ML JSON format |
| TensorFlow CSV | CSV with image paths and bbox columns |
| ImageNet Classification | Class-name subfolders |
| Cityscapes | `gtFine/` + `leftImg8bit/` structure |
| ADE20K | `images/` + `annotations/` with index files |
| DOTA | Aerial object detection with OBB |
| TFRecord | TensorFlow TFRecord binary format |

### Supported Dataset Structures

**YOLO:**
```
dataset/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

**COCO:**
```
dataset/
├── annotations/
│   ├── instances_train.json
│   └── instances_val.json
├── train/
└── val/
```

**Pascal VOC:**
```
dataset/
├── Annotations/       ← .xml files
├── ImageSets/Main/    ← train.txt, val.txt
└── JPEGImages/        ← images
```

**ImageNet-style Classification:**
```
dataset/
├── train/
│   ├── cat/
│   └── dog/
└── val/
    ├── cat/
    └── dog/
```

### Dataset Dashboard

After loading, the dashboard shows image count, total annotation objects, class distribution chart, train/val/test split breakdown, annotation coverage percentage, and detected format.

---

## Sorting & Filtering

### Basic Usage

1. Select a dataset
2. Go to **Sort & Filter**
3. Images display one at a time with annotation overlays
4. Use keyboard:
   - **← Left Arrow**: Mark for deletion
   - **→ Right Arrow**: Keep
5. Click **Apply Changes** to commit deletions

### Filtering Options

- **Annotation status**: All / annotated only / unannotated only
- **Class**: Filter to images containing a specific class

### Batch Operations

- **Shift-click**: Select a range of images
- **Delete selected**: Remove all selected images
- **Move selected**: Move to a different split (e.g. train → val)

---

## Annotation Tools

### Keyboard Shortcuts

| Tool | Shortcut |
|---|---|
| Select / Edit | V |
| Bounding Box | B |
| Polygon | P |
| Keypoint | K |
| Brush | R |
| SAM Wand | Q |
| Save | S |
| Undo | Ctrl+Z |
| Delete selected | Delete / Backspace |
| Previous image | Alt+← |
| Next image | Alt+→ |
| Close polygon | Enter |
| Cancel / Deselect | Escape |

### Bounding Box

1. Press **B**
2. Click and drag to draw
3. Release to complete
4. Select class from the right panel

### Polygon

1. Press **P**
2. Click to place each vertex
3. Press **Enter** or double-click to close
4. Select class from the right panel

### Keypoint

1. Press **K**
2. Click to place each point
3. Each point is assigned the active class label

### Brush

1. Press **R**
2. Click and drag to paint a region
3. Used for semantic segmentation masks

### SAM Wand

1. Press **Q** (or load a SAM model — the tool activates automatically)
2. Click on an object to generate a segmentation mask
3. For SAM 3: enter a text prompt as tags to guide what the model segments
4. Confirm or reject the generated mask

### Editing Annotations

1. Press **V**
2. Click an annotation to select it
3. Drag corners/edges to resize; drag center to move
4. Press **Delete** to remove

### Auto-Annotation

Run model inference across your entire dataset:

1. Ensure a model is loaded (see Model Management)
2. Click **Auto-Annotate**
3. Configure:
   - **Model**: select loaded model
   - **Confidence threshold**: minimum confidence (0.1–1.0)
   - **Text prompt** (SAM 3): a noun phrase like "person" or "yellow school bus" — SAM 3 finds every matching instance in the image
4. Click **Run**
5. Review results in Sort & Filter

---

## Class Management

### Viewing Classes

Go to **Classes** to see all classes in the current dataset with name, colour, annotation count, and image count.

### Extract Classes to a New Dataset

Pull a subset of classes into a standalone dataset:

1. Check the classes to extract
2. Click **Extract Selected**
3. Enter a name for the new dataset
4. Choose output format
5. Click **Extract**

### Delete Classes

Remove classes and all their annotations:

1. Check the classes to delete
2. Click **Delete Selected**
3. Confirm — all annotations of those classes are permanently removed

### Merge Classes

Collapse multiple class names into one:

1. Check the classes to merge (2 or more)
2. Click **Merge Selected**
3. Enter the target class name
4. Click **Merge**

Example: merging `car`, `automobile`, `vehicle` → `vehicle`

### Rename a Class

Click the edit icon next to any class name, type the new name, and press Enter.

### Add a Class

Click **+ Add Class**, enter a name, and optionally set a colour. The class is immediately available in the annotation tools.

---

## Data Augmentation

### Overview

Build an augmentation pipeline to artificially expand your dataset. Preview outputs before committing, then generate to a target image count or a multiplier of the original.

### Available Transforms

**Geometric:**

| Transform | Parameters |
|---|---|
| Horizontal Flip | probability |
| Vertical Flip | probability |
| Rotation | angle range |
| Random Scale | scale range |
| Translation | shift range |
| Shear | shear range |
| Perspective | strength |
| Random Crop | crop size range |

**Colour:**

| Transform | Parameters |
|---|---|
| Brightness | range |
| Contrast | range |
| Saturation | range |
| Hue Shift | range |
| Grayscale | probability |
| Histogram Equalisation | — |
| Channel Shuffle | probability |
| Invert | probability |
| Posterize | bits |
| Solarize | threshold |

**Noise & Blur:**

| Transform | Parameters |
|---|---|
| Gaussian Blur | kernel size |
| Gaussian Noise | mean, variance |
| Sharpen | strength |
| JPEG Compression | quality range |

**Advanced:**

| Transform | Parameters |
|---|---|
| Mosaic | combines 4 images |
| MixUp | alpha (blend ratio) |
| Cutout | hole count, size range |
| Elastic Deformation | alpha, sigma |
| Grid Distortion | steps, distort limit |

### Running Augmentation

1. Go to **Augmentation**
2. Select source dataset
3. Set target size (exact count or multiplier)
4. Toggle transforms and adjust sliders
5. Click **Preview** to see sample outputs
6. Click **Apply** to generate the augmented dataset

---

## Video Frame Extraction

### Extraction Modes

| Mode | Use case |
|---|---|
| Every Nth frame | Continuous footage with gradual change |
| Uniform distribution | Representative sample across full video |
| Keyframe detection | Videos with distinct scene cuts |
| Manual selection | Precise per-frame control |

### How to Extract

1. Go to **Video Frames**
2. Select or drop a video file
3. Choose extraction mode and configure parameters
4. Optionally set start/end time or a maximum frame count
5. Click **Extract** — a new dataset is created ready to annotate

### Supported Formats

MP4 · AVI · MOV · MKV · WebM

---

## Duplicate Detection

### Methods

| Method | What it finds | Speed |
|---|---|---|
| MD5 Hash | Exact byte-for-byte duplicates | Fastest |
| Average Hash (aHash) | Fast approximate visual similarity | Fast |
| Perceptual Hash (pHash) | Visually similar (resize-robust) | Medium |
| CLIP Embeddings | Semantically similar content | Slowest |

### How to Find and Remove Duplicates

1. Go to **Duplicates**
2. Select a dataset
3. Choose detection method
4. Set similarity threshold (higher = stricter)
5. Click **Scan**
6. Review duplicate groups
7. Choose keep strategy: **First**, **Largest resolution**, or **Smallest file size**
8. Click **Remove Duplicates**

---

## Dataset Splitting

The **Train / Val / Test** view creates train/validation/test splits from a single dataset.

### How to Split

1. Go to **Train / Val / Test**
2. Select a dataset
3. Set split ratios (default: 70 / 20 / 10)
4. Configure options:
   - **Seed**: fixed integer for reproducible splits (default: 42)
   - **Stratified**: maintain class distribution across splits
5. Enter an output name
6. Click **Split**

### Stratified Splitting

When enabled, each split preserves the class proportions of the original dataset. Example: if the source is 60% cats and 40% dogs, each of train/val/test will be approximately 60/40.

---

## Format Conversion

### How to Convert

1. Go to **Convert Format**
2. Select a source dataset (format is auto-detected)
3. Choose target format
4. Configure options:
   - Output name
   - Copy images alongside annotations, or annotations only
5. Click **Convert** — a new dataset appears in the list

### Supported Conversions

All 13 supported formats can be converted to and from each other:

YOLO ↔ YOLO OBB ↔ COCO ↔ COCO Panoptic ↔ Pascal VOC ↔ LabelMe ↔ CreateML ↔ TensorFlow CSV ↔ ImageNet ↔ Cityscapes ↔ ADE20K ↔ DOTA ↔ TFRecord

---

## Dataset Merging

### How to Merge

1. Go to **Merge**
2. Click **+ Add Dataset** and select each dataset to include
3. Resolve any class naming conflicts in the mapping table
4. Choose output name and format
5. Click **Merge**

### Class Mapping

When merging datasets that use different names for the same thing, the mapping table lets you assign each source class to a target class (or exclude it). Example:

```
Dataset A: cat, dog
Dataset B: feline, canine

Mapping:
  feline → cat
  canine → dog
```

---

## Model Management

### Available Pretrained Models

Only SAM 3 and SAM 3.1 are supported. Both are gated on HuggingFace and require a personal access token.

- **SAM 3** (`facebook/sam3`) — text-prompted concept segmentation with point and box interactive support
- **SAM 3.1** (`facebook/sam3.1`) — drop-in improvement over SAM 3 with the same API

### Downloading Models

1. Go to **Models**
2. Paste your HuggingFace access token into the input under SAM 3 / SAM 3.1 (get one at `hf.co/settings/tokens`)
3. Click the download icon — progress is shown inline
4. Once downloaded, click **Load** to make it available for inference

### Importing Custom Checkpoints

1. Go to **Models**
2. Click **Import Model**
3. Select a SAM 3 `.pt` checkpoint
4. Click **Import**

### Loading and Unloading

Click **Load** to move a model into GPU memory for inference. Click **Unload** to free that memory. Only loaded models are available for auto-annotation.

---

## Batch Jobs

The **Batch Jobs** view tracks background auto-annotation jobs. Each job shows:

- Status (running / paused / done / error)
- Per-image progress with counts of annotated vs failed
- Inline preview of recently annotated images
- Pause, resume, and cancel controls

Jobs survive page navigation and can be restored after a backend restart.

---

## Settings

### Backend Connection

Set the Python backend URL (default: `http://localhost:8000`). Click **Test Connection** to verify.

### Storage Paths

- **Models directory**: where downloaded and trained weights are saved
- **Datasets directory**: default location for loaded datasets
- **Output directory**: where converted and augmented datasets are written

### Hardware

- **GPU Device ID**: select which GPU to use if multiple are available

---

## API Reference

### Base URL

```
http://localhost:8000/api
```

Interactive Swagger docs available at `http://localhost:8000/docs` while the backend is running. No authentication required for local use.

### Datasets

```
GET    /datasets                          List all loaded datasets
POST   /datasets/load-local              Load from a local folder path
POST   /datasets/upload                  Upload a ZIP file
GET    /datasets/{id}                    Dataset details
DELETE /datasets/{id}                    Unload and delete
GET    /datasets/{id}/images             Paginated image list (?page&limit&filter)
GET    /datasets/{id}/images/{image_id}  Image with annotations
PUT    /datasets/{id}/images/{image_id}/annotations  Update annotations
```

### Class Operations

```
POST /datasets/{id}/extract-classes      Extract classes → new dataset
POST /datasets/{id}/delete-classes       Delete classes and annotations
POST /datasets/{id}/merge-classes        Merge N classes into one
```

### Augmentation

```
GET  /augmentations/list                 List available transforms
POST /datasets/{id}/augment-enhanced     Apply augmentation pipeline
```

### Video

```
POST /video/extract                      Extract frames from a video file
```

### Duplicates

```
POST /datasets/{id}/find-duplicates      Scan for duplicates
POST /datasets/{id}/remove-duplicates    Remove with a keep strategy
```

### Conversion & Merging

```
POST /datasets/{id}/convert              Convert to another format
POST /datasets/merge                     Merge multiple datasets
GET  /formats                            List supported formats
```

### Models

```
GET  /models                             List available and downloaded models
POST /models/download                    Download a pretrained model
POST /models/import                      Import a local model file
POST /models/{id}/load                   Load model into memory
POST /models/{id}/unload                 Unload model
POST /datasets/{id}/auto-annotate        Run inference on a dataset
```

### Auto-Annotation Batch Jobs

```
GET  /api/auto-annotate/jobs             List all batch annotation jobs
```

### System

```
GET /api/health                          Backend health check
```

---

## Troubleshooting

**"Backend not connected"**  
Python failed to start. Check `.logs/backend.log`. Common causes: Python < 3.10, port 8000 already in use, missing OpenCV system dependency.

**First startup hangs for a long time**  
Normal. PyTorch, transformers, and the `sam3` package are large. Watch `.logs/backend.log` for uv's install progress.

**"Dataset format not recognized"**  
Auto-detection looks for specific files (`data.yaml`, `instances_train.json`, `Annotations/*.xml`, etc). Make sure your folder structure matches exactly. Nested ZIPs inside a ZIP are not supported — extract first.

**SAM 3 download fails with 401/403**  
Both `facebook/sam3` and `facebook/sam3.1` are gated. Request access on the model page, generate a HuggingFace token at `hf.co/settings/tokens`, and paste it into the Models view on first download.

**Port 3000 or 8000 still in use after a crash**  
macOS/Linux: `lsof -ti:3000 | xargs kill -9` and `lsof -ti:8000 | xargs kill -9`  
Windows: `netstat -ano | findstr :3000` → `taskkill /F /PID <pid>`

**Blank frontend or 500 error**  
Run `npm install` manually in the project root, then `uv run app.py`. Check `.logs/frontend.log` for the cause.

**Model not loading**  
Check that the file isn't corrupted and that you have sufficient GPU memory. Try unloading other models first.

---

> ⚠️ The FastAPI backend serves files directly from your local filesystem. Do not expose port 8000 to the public internet without authentication. For remote GPU servers use SSH port forwarding: `ssh -L 3000:localhost:3000 -L 8000:localhost:8000 user@server`

---

*VisOS Documentation*