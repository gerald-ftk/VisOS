![Hero Screenshot](assets/images/hero.png)

# VisOS

> **The all-in-one local workbench for computer vision datasets.** Annotate, convert, augment, merge, and train — without touching a single cloud service or writing a line of code.

[![Build](https://img.shields.io/badge/build-passing-brightgreen?style=flat-square)](https://github.com/)
[![Version](https://img.shields.io/badge/version-0.1.0-blue?style=flat-square)](./package.json)
[![License](https://img.shields.io/badge/license-unlicensed-lightgrey?style=flat-square)](#license--credits)
[![Next.js](https://img.shields.io/badge/Next.js-16.2.0-black?style=flat-square&logo=next.js)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)](https://python.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.7-3178C6?style=flat-square&logo=typescript)](https://typescriptlang.org/)

![Hero Screenshot](assets/images/dataset_dashboard.png)
*The main dashboard after loading a dataset — showing image count, annotation coverage, class distribution chart, and train/val/test split breakdown.*

---

## Why This Exists

Managing computer vision datasets is a grind. You download a COCO dataset, realise your training framework expects YOLO format, spend an hour writing a conversion script, discover half your classes are duplicates with different names, and then have to write *another* script to clean them up. Repeat this for every project, across every team member, forever.

CV Dataset Manager puts a clean UI and a capable Python backend over all the painful, repetitive operations that sit *between* raw data and a model that actually trains. It lives entirely on your machine — no accounts, no uploads, no monthly bill.

**Who it's for:** ML engineers and researchers who work with object detection or segmentation datasets, need to wrangle multiple formats, and would rather click than script.

**What makes it different:** Most dataset tools do *one* thing — annotate, or convert, or augment. This does all of it in a single local app, with a process manager that handles startup, health-checking, and graceful shutdown automatically.

---

## Feature Overview

### 📁 Dataset Management

Load datasets from a local folder or upload a ZIP. The backend auto-detects the format on load. Loaded datasets persist across restarts via metadata sidecars.

| Supported Format | Structure Required |
|---|---|
| YOLO | `images/`, `labels/`, `data.yaml` |
| COCO | `annotations/*.json`, image folders |
| Pascal VOC | `Annotations/*.xml`, `JPEGImages/` |
| ImageNet-style Classification | `class_name/` subfolders |

![Datasets View](assets/images/datasets_view.png)
*The Datasets view showing loaded datasets with format badges, image counts, and action buttons for inspect, duplicate-scan, and delete.*

---

### 🔀 Sort & Filter

Review images one at a time with annotations overlaid. Keyboard-driven — left arrow marks for deletion, right arrow keeps. Apply bulk changes when you're done.

- Filter by annotation status (all / annotated / unannotated)
- Filter by class
- Shift-click range selection for batch delete or split-move

![Sorting View](assets/images/sort&filter.png)
*The Sort & Filter view with an image displayed, bounding box overlays visible, and the keep/delete keyboard hint at the bottom.*

---

### ✏️ Annotation Tools

A built-in annotation canvas for creating and editing labels without leaving the app.

| Tool | Shortcut | Task |
|---|---|---|
| Bounding Box | `B` | Object detection |
| Polygon | `P` | Instance segmentation |
| Point | `L` | Keypoint detection |
| Select / Edit | `V` | Move, resize, delete |

Full undo/redo history (`Ctrl+Z` / `Ctrl+Y`). Auto-annotation via loaded YOLO or RT-DETR models with configurable confidence threshold.

![Annotation View](assets/images/annotate.png)
*The annotation canvas with a bounding box being drawn over an object, the class selector panel on the right, and the undo history visible in the sidebar.*

---

### 🏷️ Class Management

Surgical operations on your class list — no manual JSON editing required.

- **Extract** — pull a subset of classes into a new standalone dataset
- **Delete** — remove classes and all their annotations in one step
- **Merge** — collapse multiple class names (e.g. `car`, `automobile`) into one
- **Rename** — inline edit of any class name

![Class Management View](assets/images/class_management.png)
*The Class Management view showing a table of classes with annotation counts, colour swatches, and checkboxes for bulk extract/delete/merge operations.*

---

### 🔄 Format Conversion

Convert any supported format to any other. Optionally copy images alongside annotations, or annotations only. Train/val/test splits can be created during conversion with configurable ratios and optional stratification by class.

![Convert View](assets/images/convert_format.png)
*The Convert view with source format auto-detected as COCO, target format set to YOLO, and split ratio sliders set to 70/20/10.*

---

### 🧬 Data Augmentation

Build augmentation pipelines with a toggle-based UI, preview examples before committing, and output a new dataset at a target size or multiplier.

**Geometric:** flip (H/V), rotation, random crop, resize, perspective  
**Colour:** brightness, contrast, saturation, hue shift  
**Noise/Blur:** Gaussian blur, motion blur, Gaussian noise  
**Advanced:** mosaic (4-image combine), MixUp, CutOut  
**Presets:** Light / Medium / Heavy — one click to configure common stacks

![Augmentation View](assets/images/augmentation.png)
*The Augmentation view with a pipeline configured — several transforms enabled with parameter sliders, the preview pane showing four augmented sample images.*

---

### 🎬 Video Frame Extraction

Turn video files into image datasets without external tools.

- **Every Nth frame** — e.g. every 30th frame = 1 fps from 30 fps footage
- **Uniform distribution** — exactly N frames spread across the full video
- **Keyframe detection** — extract on scene changes
- **Manual selection** — scrub and mark individual frames

Supports MP4, AVI, MOV, MKV, WebM. Output is a new dataset ready to annotate.

![Video Extraction View](assets/images/frame_extraction.png)
*The Video Extraction view with a video loaded in the preview player, frame interval set to 15, and an estimated frame count shown before extraction starts.*

---

### 🔍 Duplicate Detection

Find and remove exact or near-duplicate images from a dataset.

| Method | What it finds |
|---|---|
| MD5 Hash | Exact byte-for-byte duplicates |
| Perceptual Hash (pHash) | Visually similar images (resize-robust) |
| Average Hash (aHash) | Fast approximate similarity |
| CLIP Embeddings | Semantically similar content (AI-powered) |

Configurable similarity threshold. Choose a keep strategy — first, largest resolution, or smallest file — then remove in one click.

---

### 🔗 Dataset Merging

Combine multiple datasets into one, with a class-mapping UI to resolve naming conflicts between sources before merging.

![Merge View](assets/images/merge_datasets.png)
*The Merge view with two datasets added, a class conflict resolved via a mapping table (feline → cat, canine → dog), and the output format selector.*

---

### 🤖 Model Management

Download pretrained models directly from the app or import your own `.pt`, `.pth`, or `.onnx` files. Load/unload models to manage GPU memory.

**Available for download:**
- YOLOv8 n / s / m / l / x
- SAM ViT-B / ViT-L / ViT-H
- RT-DETR L / X

Loaded models become available for auto-annotation.

---

### 🏋️ Training

Train object detection and segmentation models locally with real-time metric monitoring.

- Architectures: YOLOv8 (n/s/m/l/x), YOLOv9 (c/e), RT-DETR (l/x)
- Configurable: epochs, batch size, image size, learning rate, early-stopping patience
- Live display of loss, accuracy, validation loss, GPU usage, and ETA
- Pause, resume, and stop with checkpoint saving
- Export trained weights to PyTorch, ONNX, or TensorRT

![Training View](assets/images/training_view.png)
*The Training view mid-run — loss and accuracy charts updating live, GPU usage percentage shown, current epoch highlighted, and Pause/Stop controls visible.*

---

### 📸 Image Gallery

Browse all images in a dataset as a grid with annotation overlays, click through to full-size view with class labels.

### 📊 Dataset Comparison

Side-by-side stats comparison between two loaded datasets — class distributions, annotation counts, image dimensions.

### 📷 Dataset Snapshots

Save a named snapshot of a dataset's current state, and restore to it later. Useful before destructive operations like class deletion or augmentation.

### 📝 YAML Wizard

GUI for creating and editing `data.yaml` files for YOLO-format datasets — no manual editing required.

### ❤️ Health View

Real-time system health panel showing backend API status, Python dependencies, GPU availability, and workspace directory usage.

---

## Demo

![Demo](docs/screenshots/demo.gif)
*A full walkthrough: loading a YOLO dataset → inspecting the dashboard → converting to COCO → applying augmentation → starting a YOLOv8n training run and watching metrics update.*

---

## Getting Started

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | Must be on `PATH` |
| Node.js | 18+ | LTS recommended |
| npm or pnpm | any | pnpm preferred |
| Git | any | For cloning |
| NVIDIA GPU | optional | For training; CPU works for annotation and conversion |

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-org/cv-dataset-manager.git
cd cv-dataset-manager
```

**2. Start the app**

```bash
# macOS / Linux
python3 run.py start

# Windows
python run.py start
```

That's it. The process manager:
- Creates a Python virtual environment in `backend/venv/`
- Installs Python dependencies from `backend/requirements.txt`
- Installs Node.js packages (npm/pnpm)
- Starts the FastAPI backend on port `8000`
- Starts the Next.js frontend on port `3000`
- Health-checks both services and opens your browser

**First-time startup takes 2–5 minutes** while Python packages (Pillow, OpenCV, Ultralytics, etc.) download.

### Quick Start

```bash
# Start everything
python3 run.py start

# Open http://localhost:3000
# Go to Settings → verify "Connected" status
# Go to Datasets → Open Local Folder → select your dataset folder
```

### Environment Variables

There are no required `.env` variables for local use. The backend URL is configured through the Settings UI and defaults to `http://localhost:8000`.

If you need to override ports, edit the relevant lines in `run.py`:

```python
# Backend port (run.py, start_backend())
"--port", "8000",

# Frontend port — set via npm/next dev
# Add --port 3001 to the frontend start command if needed
```

---

## Usage Guide

### Starting and stopping

```bash
python3 run.py start          # Start both servers
python3 run.py stop           # Stop both servers cleanly
python3 run.py restart        # Full stop → start
python3 run.py restart-back   # Restart backend only (after Python changes)
python3 run.py restart-front  # Restart frontend only
python3 run.py status         # Show running PIDs and ports
python3 run.py logs           # Tail live output from both servers
```

### Loading your first dataset

1. Open `http://localhost:3000`
2. Navigate to **Datasets** in the sidebar
3. Click **Open Local Folder** and browse to your dataset directory
4. The backend auto-detects the format — no configuration needed
5. Click **Load** — the dataset appears in the list with stats

![Datasets Loaded](docs/screenshots/datasets-loaded.png)
*The Datasets view after loading two datasets, showing format badges (YOLO, COCO), image counts, and action icons.*

### Annotating images

1. Select a dataset, then go to **Annotations** in the sidebar
2. Press `B` to activate the bounding box tool
3. Click and drag to draw a box — release to complete
4. Select the class from the right panel
5. Press `Ctrl+Z` to undo if needed
6. Annotations save automatically

### Converting a dataset

1. Go to **Convert** in the sidebar
2. Select the source dataset
3. Choose the target format (YOLO, COCO, VOC, CSV, JSON)
4. Set an output name and optional split ratios
5. Click **Convert** — a new dataset appears in your list

### Running augmentation

1. Go to **Augmentation**
2. Select the source dataset
3. Set the target size (e.g. `5000` images or `3x` multiplier)
4. Toggle the augmentations you want and adjust sliders
5. Click **Preview** to see sample outputs
6. Click **Apply** to generate the augmented dataset

### Training a model

1. Go to **Training**
2. Select a dataset (must be YOLO format or compatible)
3. Choose architecture — `yolov8n` is fastest for testing
4. Set epochs, batch size, and image size
5. Click **Start Training**
6. Watch loss/accuracy charts update live

---

## Architecture & Project Structure

```
cv-dataset-manager/
├── run.py                    # Cross-platform process manager (start/stop/restart/logs)
├── start.sh                  # Shell shortcut → delegates to run.py
├── start.bat                 # Windows shortcut → delegates to run.py
│
├── backend/                  # Python FastAPI backend
│   ├── main.py               # App entrypoint, all API routes, workspace management
│   ├── dataset_parsers.py    # Auto-detect & parse YOLO / COCO / VOC / classification
│   ├── format_converter.py   # Convert between any two supported formats
│   ├── annotation_tools.py   # Annotation read/write/update logic
│   ├── augmentation.py       # Augmentation pipeline engine
│   ├── dataset_merger.py     # Multi-dataset merge with class mapping
│   ├── model_integration.py  # Model download, load/unload, inference (YOLO, SAM, RT-DETR)
│   ├── training.py           # Training job management, real-time metric streaming
│   ├── video_utils.py        # Frame extraction, duplicate detection, CLIP embeddings
│   └── requirements.txt      # Python dependencies
│
├── app/                      # Next.js App Router pages
│   ├── layout.tsx            # Root layout with theme provider and sidebar
│   ├── page.tsx              # Root redirect to dashboard
│   ├── globals.css           # Global styles
│   └── api/
│       ├── backend/route.ts  # Proxy: forwards requests to FastAPI backend
│       └── formats/route.ts  # Returns supported format list
│
├── components/               # React view components (one per sidebar section)
│   ├── dashboard-view.tsx    # Dataset stats, charts, overview
│   ├── datasets-view.tsx     # Dataset list, load/upload/delete
│   ├── annotation-view.tsx   # Canvas-based annotation editor
│   ├── sorting-view.tsx      # Image-by-image review with keep/delete
│   ├── gallery-view.tsx      # Grid image browser
│   ├── augmentation-view.tsx # Augmentation pipeline builder
│   ├── convert-view.tsx      # Format conversion + split UI
│   ├── merge-view.tsx        # Multi-dataset merge with class mapping
│   ├── class-management-view.tsx  # Extract / delete / merge / rename classes
│   ├── models-view.tsx       # Model download, import, load/unload
│   ├── training-view.tsx     # Training config and live metrics
│   ├── video-extraction-view.tsx  # Video → frames extraction
│   ├── compare-view.tsx      # Side-by-side dataset comparison
│   ├── snapshot-view.tsx     # Dataset snapshot save/restore
│   ├── health-view.tsx       # Backend health and system status
│   ├── settings-view.tsx     # Backend URL, paths, hardware, preferences
│   ├── yaml-wizard-view.tsx  # data.yaml GUI editor
│   └── sidebar.tsx           # Navigation sidebar
│
├── components/ui/            # shadcn/ui component library (Radix-based)
├── lib/
│   ├── types.ts              # Shared TypeScript types
│   ├── utils.ts              # cn() utility and helpers
│   └── settings-context.tsx  # Global settings state (backend URL, paths)
├── hooks/                    # use-mobile, use-toast
└── scripts/
    ├── run_backend.sh        # Direct backend start (bypasses process manager)
    └── run_backend.bat       # Windows equivalent
```

### Tech Stack

| Technology | Purpose | Version |
|---|---|---|
| Next.js | Frontend framework (App Router) | 16.2.0 |
| React | UI rendering | 19 |
| TypeScript | Type-safe frontend code | 5.7.3 |
| Tailwind CSS | Utility-first styling | 4.2.0 |
| shadcn/ui + Radix UI | Accessible component primitives | various |
| Recharts | Training metric charts | 2.15.0 |
| FastAPI | Python backend API | ≥ 0.109 |
| Uvicorn | ASGI server for FastAPI | ≥ 0.27 |
| OpenCV | Image processing, video frame extraction | ≥ 4.8 |
| Pillow | Image I/O and augmentation | ≥ 10.0 |
| Ultralytics | YOLO model inference and training | ≥ 8.1 |
| PyYAML | YAML parsing for dataset configs | ≥ 6.0 |
| psutil | System/GPU monitoring | ≥ 5.9 |

### Key architectural decisions

- **Proxy pattern**: The Next.js API routes in `app/api/backend/route.ts` forward requests to the FastAPI backend. This avoids CORS issues and means the frontend only ever talks to `localhost:3000`.
- **Workspace persistence**: Datasets survive restarts via `dataset_metadata.json` sidecar files written alongside each dataset. On startup, the backend scans `workspace/datasets/` and re-registers everything it finds.
- **Process manager in Python**: `run.py` manages both processes, handles cross-platform PID tracking, streams logs from both servers to the terminal simultaneously, and cleans up ports on exit — without requiring Docker.

---

## Configuration

### `next.config.mjs`

```js
const nextConfig = {
  typescript: { ignoreBuildErrors: true },  // allows incremental typing
  images: { unoptimized: true },            // avoids Next.js image API for local files
}
```

### `backend/requirements.txt`

Core dependencies install automatically. Optional heavy dependencies are commented out:

```
# ultralytics >= 8.1.0   — YOLO (installs torch automatically)
# segment-anything        — SAM (optional, install manually if needed)
# albumentations          — faster augmentation (optional)
```

### Workspace layout

The backend creates this structure on first run:

```
workspace/
├── datasets/     # Loaded datasets (each in its own UUID subfolder)
├── models/       # Downloaded and imported model weights
├── exports/      # Converted/exported datasets
├── snapshots/    # Dataset snapshots
└── temp/         # Temporary files during processing
```

### `package.json` scripts

```bash
npm run dev     # Start Next.js in dev mode (HMR enabled)
npm run build   # Production build
npm run start   # Serve production build
npm run lint    # ESLint
```

---

## API Reference

Base URL: `http://localhost:8000/api`  
Interactive docs: `http://localhost:8000/docs` (Swagger UI, auto-generated)

Authentication: none required for local use.

### Datasets

```
GET    /datasets                          List all loaded datasets
POST   /datasets/load-local              Load a dataset from a local folder path
POST   /datasets/upload                  Upload a ZIP file as a new dataset
GET    /datasets/{id}                    Get dataset details
DELETE /datasets/{id}                    Unload and delete a dataset
GET    /datasets/{id}/images             Paginated image list (?page&limit&filter)
GET    /datasets/{id}/images/{image_id}  Single image with annotations
PUT    /datasets/{id}/images/{image_id}/annotations  Update annotations
```

### Class operations

```
POST /datasets/{id}/extract-classes      Extract classes → new dataset
POST /datasets/{id}/delete-classes       Delete classes and their annotations
POST /datasets/{id}/merge-classes        Merge N classes into one
```

### Augmentation

```
GET  /augmentations/list                 List available augmentation types
POST /datasets/{id}/augment-enhanced     Apply augmentation pipeline
```

### Video

```
POST /video/extract                      Extract frames from a video file
```

### Duplicates

```
POST /datasets/{id}/find-duplicates      Scan for duplicates (returns groups)
POST /datasets/{id}/remove-duplicates    Remove duplicates with a keep strategy
```

### Conversion & merging

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

### Training

```
POST /training/start                     Start a training job
GET  /training/{job_id}/status           Get metrics and status
POST /training/{job_id}/pause            Pause training
POST /training/{job_id}/resume           Resume training
POST /training/{job_id}/stop             Stop and save checkpoint
```

### System

```
GET /api/health                          Backend health check
```

---

## Development Guide

### Running in dev mode

```bash
# Start both with hot reload
python3 run.py start

# Backend hot-reloads on Python file changes (uvicorn --reload)
# Frontend hot-reloads on TypeScript/React changes (Next.js HMR)

# Restart just the backend after changes that break the reload
python3 run.py restart-back
```

### Tail live logs

```bash
python3 run.py logs
```

### Building for production

```bash
# Build the Next.js frontend
npm run build

# Then serve it alongside the backend
npm run start        # Frontend on :3000
# In another terminal:
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000
```

### Linting

```bash
npm run lint         # ESLint on all TypeScript files
```

### Adding a new view / feature

1. **Backend**: Add a route (or routes) to `backend/main.py`. Put heavy logic in a dedicated module (e.g. `backend/my_feature.py`) and import it in `main.py`.
2. **Frontend**: Create `components/my-feature-view.tsx`. The component fetches from `/api/backend/...` (the Next.js proxy) rather than directly from `:8000`.
3. **Navigation**: Add an entry to the sidebar in `components/sidebar.tsx` and handle the new view key in `app/page.tsx` or the root layout's view-switcher.
4. **Types**: Add any shared types to `lib/types.ts`.

### Contributing

- Branch naming: `feat/short-description`, `fix/short-description`, `chore/short-description`
- Keep PRs focused on one concern
- Run `npm run lint` before opening a PR
- The backend has no test suite yet — manual testing against a real dataset is the current workflow

---

## Deployment

This application is designed for **local use**. There is no Dockerfile, no CI/CD config, and no production deployment configuration in the repository.

If you want to run it on a remote machine (e.g. a GPU server):

**1. SSH with port forwarding**

```bash
ssh -L 3000:localhost:3000 -L 8000:localhost:8000 user@your-server
# Then start the app on the server as normal
python3 run.py start
# Access http://localhost:3000 in your local browser
```

**2. Running without the process manager**

```bash
# Terminal 1 — backend
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — frontend
npm install
npm run dev
```

> ⚠️ **Note**: The FastAPI backend serves files from the local filesystem. Exposing it to the public internet without authentication would allow anyone to read and delete files on your server. Only expose it behind a VPN or SSH tunnel.

---

## Roadmap

Based on code structure, commented TODOs, and optional dependencies referenced but not yet fully integrated:

- [ ] **Albumentations integration** — faster, GPU-accelerated augmentation pipeline as an alternative to the current PIL/OpenCV implementation (`albumentations` is listed in `requirements.txt` as optional)
- [ ] **SAM 2 auto-annotation** — Segment Anything Model 2 is referenced in the model list but marked as optional; full polygon auto-annotation from a bounding box prompt
- [ ] **Training test suite** — the backend currently has no automated tests; adding pytest coverage for parsers and converters would significantly reduce regression risk
- [ ] **Dataset versioning** — the snapshot system exists but a full git-style diff between snapshots would make iterative cleaning workflows safer
- [ ] **CLIP-powered image search** — CLIP embeddings are implemented in `video_utils.py` for duplicate detection; the same infrastructure could power semantic image search within a dataset
- [ ] **Collaborative annotation** — the current architecture is single-user; multi-user support would require moving the workspace to a shared path and adding locking

---

## Troubleshooting

**"Backend not connected" shown in Settings**  
The Python backend failed to start. Check `.logs/backend.log` for the error. Common causes: Python < 3.10, a missing system dependency for OpenCV, or port 8000 already in use. Run `python3 run.py status` to check.

**First startup hangs for a long time**  
This is normal. Ultralytics and its PyTorch dependency are large (~1.5 GB) and download on first run. Check `.logs/backend.log` to see pip's progress.

**"Dataset format not recognized"**  
The auto-detection checks for specific files (`data.yaml`, `instances_train.json`, `.xml` in `Annotations/`). Make sure your folder structure matches one of the supported layouts exactly. Nested ZIPs inside a ZIP are not supported — extract first.

**Out of memory during training**  
Reduce batch size (try 4 or 8), reduce image size (try 320 or 416), or switch to a smaller architecture (yolov8n instead of yolov8l). Close other GPU-using applications. Check GPU VRAM with `nvidia-smi`.

**Port 3000 or 8000 already in use after a crash**  
`run.py stop` cleans up ports, but if it was killed forcefully: `lsof -ti:3000 | xargs kill -9` and `lsof -ti:8000 | xargs kill -9` on macOS/Linux. On Windows: `netstat -ano | findstr :3000` then `taskkill /F /PID <pid>`.

**Frontend shows blank page or 500 error**  
Check `.logs/frontend.log`. Usually caused by a missing `node_modules` — run `npm install` manually in the project root, then `python3 run.py restart-front`.

---

## License & Credits

This project is currently unlicensed — all rights reserved by the author unless otherwise stated.

### Key dependencies

| Package | What it contributes |
|---|---|
| [Ultralytics](https://github.com/ultralytics/ultralytics) | YOLO model training and inference |
| [FastAPI](https://fastapi.tiangolo.com/) | Python backend API framework |
| [OpenCV](https://opencv.org/) | Image processing and video frame extraction |
| [Next.js](https://nextjs.org/) | React frontend framework |
| [shadcn/ui](https://ui.shadcn.com/) | Accessible UI component system |
| [Recharts](https://recharts.org/) | Training metric visualisation |
| [Pillow](https://python-pillow.org/) | Image augmentation and I/O |

### Author

See repository contributors.  
API documentation available locally at `http://localhost:8000/docs` once the backend is running.