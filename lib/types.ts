// Dataset Types
export type DatasetFormat = 
  | "yolo"
  | "yolo_seg"
  | "coco"
  | "coco_seg"
  | "pascal_voc"
  | "csv"
  | "txt"
  | "json"
  | "classification"
  | "labelme"
  | "cvat"

export type TaskType = 
  | "detection"
  | "segmentation"
  | "classification"
  | "instance_segmentation"
  | "semantic_segmentation"
  | "pose"

export interface BoundingBox {
  x: number
  y: number
  width: number
  height: number
}

export interface Polygon {
  points: [number, number][]
}

export interface Annotation {
  id: string
  classId: number
  className: string
  type: "bbox" | "polygon" | "point" | "classification"
  bbox?: BoundingBox
  polygon?: Polygon
  confidence?: number
  isManual?: boolean
}

export interface DatasetImage {
  id: string
  filename: string
  path: string
  width: number
  height: number
  annotations: Annotation[]
  status: "annotated" | "unannotated" | "needs_review"
}

export interface DatasetClass {
  id: number
  name: string
  color: string
  count: number
}

export interface Dataset {
  id: string
  name: string
  path: string
  format: DatasetFormat
  taskType: TaskType
  images: DatasetImage[]
  classes: DatasetClass[]
  totalImages: number
  annotatedImages: number
  createdAt: string
  updatedAt: string
}

// Model Types
export type ModelType = "sam3"

export interface Model {
  id: string
  name: string
  type: ModelType
  task: TaskType
  path: string
  size: string
  params: string
  loaded: boolean
  accuracy?: number
  addedAt: string
}

// Merge Types
export interface MergeConfig {
  datasets: string[]
  outputFormat: DatasetFormat
  outputPath: string
  outputName: string
  remapClasses: boolean
  classMapping?: Record<string, string>
}

// Convert Types
export interface ConvertConfig {
  inputPath: string
  inputFormat: DatasetFormat
  outputFormat: DatasetFormat
  outputPath: string
  splitRatio?: {
    train: number
    val: number
    test: number
  }
}

// Auto-annotation Types
export interface AutoAnnotateConfig {
  datasetPath: string
  modelId: string
  confidenceThreshold: number
  iouThreshold: number
  batchSize: number
  device: "cpu" | "cuda" | "mps"
}

// API Response Types
export interface ApiResponse<T> {
  success: boolean
  data?: T
  error?: string
  message?: string
}

export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  pageSize: number
  totalPages: number
}
