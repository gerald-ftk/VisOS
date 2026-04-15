"use client"
// ── CHANGES vs PREVIOUS VERSION ───────────────────────────────────────────────
// • filteredLoaded filter: guarded m.name with `(m.name ?? "")` to prevent
//   TypeError crash when backend returns a model with name: null/undefined.
// ──────────────────────────────────────────────────────────────────────────────

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { 
  Upload, 
  Download,
  Trash2,
  Search,
  Box,
  Cpu,
  FileCode,
  CheckCircle2,
  HardDrive,
  MoreVertical,
  RefreshCw,
  AlertCircle
} from "lucide-react"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

// Shape returned by GET /api/models
interface BackendModel {
  id: string
  name: string
  type: string        // "sam3"
  loaded: boolean
  pretrained?: boolean
  downloaded?: boolean
  path?: string
  classes?: string[]
}

interface ModelsViewProps {
  apiUrl?: string
}

// SAM 3 / 3.1 — the only models the backend supports. Both are gated on HF.
const PRETRAINED_CATALOG: Array<{
  id: string
  name: string
  type: string
  task: string
  sizeLabel: string
  requiresHfToken?: boolean
}> = [
  { id: "sam3",  name: "SAM 3",   type: "sam3", task: "segmentation", sizeLabel: "~3.5 GB", requiresHfToken: true },
  { id: "sam31", name: "SAM 3.1", type: "sam3", task: "segmentation", sizeLabel: "~3.5 GB", requiresHfToken: true },
]

// Shared localStorage key for the HuggingFace token so both the Models and
// Annotate views can reuse it when downloading gated checkpoints.
const HF_TOKEN_KEY = "opensamannotator.hf_token"

export function ModelsView({ apiUrl = "http://localhost:8000" }: ModelsViewProps) {
  const [models, setModels] = useState<BackendModel[]>([])
  const [searchQuery, setSearchQuery] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [loadingModelId, setLoadingModelId] = useState<string | null>(null)
  const [downloadingIds, setDownloadingIds] = useState<Set<string>>(new Set())
  const [downloadProgress, setDownloadProgress] = useState<Record<string, number>>({})
  const [error, setError] = useState<string | null>(null)
  // Hydrate the HF token from localStorage so users only have to paste it
  // once, and so the Annotate view can reuse it for gated auto-downloads.
  const [hfToken, setHfToken] = useState<string>(() => {
    if (typeof window === "undefined") return ""
    try { return window.localStorage.getItem(HF_TOKEN_KEY) ?? "" } catch { return "" }
  })
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (typeof window === "undefined") return
    try {
      if (hfToken) window.localStorage.setItem(HF_TOKEN_KEY, hfToken)
      else window.localStorage.removeItem(HF_TOKEN_KEY)
    } catch {}
  }, [hfToken])

  // -------------------------------------------------------------------------
  // Fetch models from backend on mount
  // -------------------------------------------------------------------------
  const fetchModels = async () => {
    setIsLoading(true)
    setError(null)
    try {
      const res = await fetch(`${apiUrl}/api/models`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setModels(data.models ?? [])
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => { fetchModels() }, [apiUrl])

  // -------------------------------------------------------------------------
  // Load a model via the backend
  // -------------------------------------------------------------------------
  const handleLoadModel = async (model: BackendModel) => {
    setLoadingModelId(model.id)
    try {
      const formData = new FormData()
      formData.append("model_type", model.type)
      if (model.path) formData.append("model_name", model.name)
      if (model.pretrained) formData.append("pretrained", model.id)

      const res = await fetch(`${apiUrl}/api/models/load`, {
        method: "POST",
        body: formData,
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(err.detail ?? res.statusText)
      }
      await fetchModels()
    } catch (err) {
      setError(`Load failed: ${err instanceof Error ? err.message : err}`)
    } finally {
      setLoadingModelId(null)
    }
  }

  // Ask the backend to drop the model from its in-memory registry, then
  // refetch. This also clears any stale "failed load" entry (e.g. a gated
  // HF model attempted without a token) so the token input reappears.
  const handleUnloadModel = async (model: BackendModel) => {
    setModels(prev => prev.map(m => m.id === model.id ? { ...m, loaded: false } : m))
    try {
      await fetch(`${apiUrl}/api/models/${encodeURIComponent(model.id)}/unload`, { method: "POST" })
    } catch {
      // Network hiccup — the optimistic update above still leaves the UI
      // in a sensible state; the next fetchModels() will reconcile.
    } finally {
      await fetchModels()
    }
  }

  // Backend has no delete endpoint for loaded models — remove from local list
  const handleDeleteModel = (modelId: string) => {
    setModels(prev => prev.filter(m => m.id !== modelId))
  }

  // -------------------------------------------------------------------------
  // Download (load pretrained) via the backend
  // -------------------------------------------------------------------------
  const handleDownloadPreset = async (preset: typeof PRETRAINED_CATALOG[0]) => {
    setDownloadingIds(prev => new Set([...prev, preset.id]))
    setDownloadProgress(prev => ({ ...prev, [preset.id]: 5 }))
    setError(null)

    try {
      // Start background download
      const formData = new FormData()
      formData.append("model_type", preset.type)
      if (preset.requiresHfToken && hfToken) formData.append("hf_token", hfToken)
      formData.append("pretrained", preset.id)
      const startRes = await fetch(`${apiUrl}/api/models/download`, { method: "POST", body: formData })
      if (!startRes.ok) {
        const err = await startRes.json().catch(() => ({ detail: startRes.statusText }))
        throw new Error(err.detail ?? startRes.statusText)
      }

      // Poll for real progress
      await new Promise<void>((resolve, reject) => {
        const poll = setInterval(async () => {
          try {
            const statusRes = await fetch(`${apiUrl}/api/models/download-status/${preset.id}`)
            if (!statusRes.ok) return
            const status = await statusRes.json()
            if (status.status === "done") {
              setDownloadProgress(prev => ({ ...prev, [preset.id]: 100 }))
              clearInterval(poll)
              resolve()
            } else if (status.status === "error") {
              clearInterval(poll)
              reject(new Error(status.error ?? "Download failed"))
            } else {
              setDownloadProgress(prev => ({
                ...prev,
                [preset.id]: Math.min((prev[preset.id] ?? 5) + 3, 90)
              }))
            }
          } catch { /* network blip, keep polling */ }
        }, 1000)
      })

      await fetchModels()
    } catch (err) {
      setError(`Download failed: ${err instanceof Error ? err.message : err}`)
    } finally {
      setDownloadingIds(prev => { const n = new Set(prev); n.delete(preset.id); return n })
      setDownloadProgress(prev => { const n = { ...prev }; delete n[preset.id]; return n })
    }
  }

  // -------------------------------------------------------------------------
  // Import custom model file
  // -------------------------------------------------------------------------
  const handleImportModel = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setLoadingModelId("import")
    setError(null)
    try {
      const formData = new FormData()
      formData.append("model_file", file)
      formData.append("model_type", "sam3")
      formData.append("model_name", file.name)

      const res = await fetch(`${apiUrl}/api/models/load`, {
        method: "POST",
        body: formData,
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(err.detail ?? res.statusText)
      }
      await fetchModels()
    } catch (err) {
      setError(`Import failed: ${err instanceof Error ? err.message : err}`)
    } finally {
      setLoadingModelId(null)
      // Reset input so same file can be re-selected
      if (fileInputRef.current) fileInputRef.current.value = ""
    }
  }

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------
  const getTypeColor = (type: string) => {
    switch (type) {
      case "sam3": return "bg-pink-500/10 text-pink-500"
      default:     return "bg-orange-500/10 text-orange-500"
    }
  }

  const TypeIcon = ({ type }: { type: string }) => {
    if (type === "sam3") return <FileCode className="w-6 h-6" />
    return <Box className="w-6 h-6" />
  }

  // "Your Models" = anything loaded in memory OR downloaded to disk (pretrained
  // or not). Exclude entries whose only reason for being in the list is a
  // previous failed-load error — they have no usable weights.
  const loadedModels = models.filter(
    m => !((m as any).error) && (m.loaded || (m as any).downloaded)
  )

  const filteredLoaded = loadedModels.filter(m =>
    (m.name ?? "").toLowerCase().includes(searchQuery.toLowerCase())
  )

  // Which pretrained IDs are downloaded (on disk) or loaded in memory?
  // Same exclusion: an entry carrying an error doesn't count as downloaded.
  const downloadedPretrainedIds = new Set(
    models
      .filter(m => m.pretrained && !((m as any).error) && (m.loaded || (m as any).downloaded))
      .map(m => m.id)
  )

  return (
    <div className="p-6 space-y-6">
      {/* Hidden file input for import */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".pt,.pth"
        className="hidden"
        onChange={handleImportModel}
      />

      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-foreground">Models</h1>
          <p className="text-muted-foreground mt-1">Download and manage SAM 3 / SAM 3.1 for auto-annotation</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="icon" onClick={fetchModels} disabled={isLoading}>
            <RefreshCw className={`w-4 h-4 ${isLoading ? "animate-spin" : ""}`} />
          </Button>
          <Button className="gap-2" onClick={() => fileInputRef.current?.click()} disabled={loadingModelId === "import"}>
            <Upload className="w-4 h-4" />
            {loadingModelId === "import" ? "Importing…" : "Import Model"}
          </Button>
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-2 rounded-md bg-destructive/10 border border-destructive/30 p-3 text-sm text-destructive">
          <AlertCircle className="w-4 h-4 shrink-0" />
          {error}
          <button className="ml-auto text-xs underline" onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}

      <div className="relative max-w-md">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
        <Input
          placeholder="Search models..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="pl-10"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Your Models */}
        <div className="lg:col-span-2">
          <Card className="border-border/50 bg-card/50 backdrop-blur">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Box className="w-5 h-5 text-primary" />
                Your Models
              </CardTitle>
              <CardDescription>Loaded and imported models available for auto-annotation</CardDescription>
            </CardHeader>
            <CardContent>
              {isLoading && filteredLoaded.length === 0 ? (
                <div className="flex items-center justify-center py-12 text-muted-foreground gap-2">
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Loading models…
                </div>
              ) : filteredLoaded.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-12 text-center">
                  <div className="w-16 h-16 rounded-full bg-muted/50 flex items-center justify-center mb-4">
                    <Box className="w-8 h-8 text-muted-foreground" />
                  </div>
                  <p className="text-muted-foreground mb-4">No models loaded yet</p>
                  <p className="text-sm text-muted-foreground/70 mb-4">
                    Download SAM 3 / SAM 3.1 or import a local checkpoint
                  </p>
                  <Button variant="outline" className="gap-2" onClick={() => fileInputRef.current?.click()}>
                    <Upload className="w-4 h-4" />
                    Import a Model
                  </Button>
                </div>
              ) : (
                <div className="space-y-3">
                  {filteredLoaded.map(model => (
                    <div
                      key={model.id}
                      className={`flex items-center gap-4 p-4 rounded-lg border transition-all ${
                        model.loaded ? "border-primary/50 bg-primary/5" : "border-border/50 bg-muted/20"
                      }`}
                    >
                      <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${getTypeColor(model.type)}`}>
                        <TypeIcon type={model.type} />
                      </div>

                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="font-medium">{model.name}</span>
                          {model.loaded && (
                            <Badge className="text-xs bg-green-500/10 text-green-500">
                              <CheckCircle2 className="w-3 h-3 mr-1" />
                              Loaded
                            </Badge>
                          )}
                        </div>
                        <div className="flex items-center gap-3 text-xs text-muted-foreground">
                          <Badge variant="outline" className={getTypeColor(model.type)}>
                            {model.type.toUpperCase()}
                          </Badge>
                          {model.path && (
                            <span className="flex items-center gap-1 truncate max-w-[200px]">
                              <HardDrive className="w-3 h-3 shrink-0" />
                              {model.path}
                            </span>
                          )}
                        </div>
                      </div>

                      <div className="flex items-center gap-2">
                        {model.loaded ? (
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleUnloadModel(model)}
                            disabled={loadingModelId === model.id}
                          >
                            Unload
                          </Button>
                        ) : (
                          <Button
                            size="sm"
                            onClick={() => handleLoadModel(model)}
                            disabled={loadingModelId === model.id}
                          >
                            {loadingModelId === model.id ? (
                              <RefreshCw className="w-3 h-3 animate-spin" />
                            ) : "Load"}
                          </Button>
                        )}

                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="icon">
                              <MoreVertical className="w-4 h-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuSeparator />
                            <DropdownMenuItem
                              className="text-destructive"
                              onClick={() => handleDeleteModel(model.id)}
                            >
                              <Trash2 className="w-4 h-4 mr-2" />
                              Remove
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Download pretrained models */}
        <Card className="border-border/50 bg-card/50 backdrop-blur">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Download className="w-5 h-5 text-primary" />
              Download Models
            </CardTitle>
            <CardDescription>SAM 3 / SAM 3.1 — gated on HuggingFace, token required</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 max-h-[520px] overflow-auto">
              {PRETRAINED_CATALOG.map(preset => {
                const isDownloading = downloadingIds.has(preset.id)
                const isDownloaded = downloadedPretrainedIds.has(preset.id)
                const backendEntry = models.find(m => m.id === preset.id)
                const isLoaded = !!backendEntry?.loaded
                const hasError = !!(backendEntry as any)?.error
                const progress = downloadProgress[preset.id] ?? 0

                // Show the token input whenever a gated model isn't actually
                // usable yet — i.e. not downloaded, not loaded, OR carrying a
                // previous failed-load error from the backend.
                const needsToken = preset.requiresHfToken && (!isDownloaded || hasError) && !isLoaded

                return (
                  <div
                    key={preset.id}
                    className="flex flex-col gap-2 p-3 rounded-lg border border-border/50 bg-muted/20 hover:bg-muted/30 transition-colors"
                  >
                    <div className="flex items-center gap-3">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <p className="font-medium text-sm">{preset.name}</p>
                          {preset.requiresHfToken && (
                            <Badge variant="outline" className="text-[10px] text-amber-500 border-amber-500/50 shrink-0">HF Auth</Badge>
                          )}
                        </div>
                        <div className="flex items-center gap-2 text-xs text-muted-foreground">
                          <span>{preset.sizeLabel}</span>
                          <span className="text-muted-foreground/50">|</span>
                          <span>{preset.task}</span>
                        </div>
                        {isDownloading && (
                          <Progress value={progress} className="h-1 mt-2" />
                        )}
                      </div>

                      {isLoaded ? (
                        <Badge variant="secondary" className="text-xs shrink-0">
                          <CheckCircle2 className="w-3 h-3 mr-1" />
                          Loaded
                        </Badge>
                      ) : isDownloading ? (
                        <RefreshCw className="w-4 h-4 animate-spin text-muted-foreground shrink-0" />
                      ) : isDownloaded ? (
                        <Badge variant="outline" className="text-xs shrink-0 text-emerald-600 border-emerald-600">
                          <CheckCircle2 className="w-3 h-3 mr-1" />
                          Downloaded
                        </Badge>
                      ) : (
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8 shrink-0"
                          onClick={() => handleDownloadPreset(preset)}
                          disabled={needsToken && !hfToken.trim()}
                          title={needsToken && !hfToken.trim() ? "Paste your HuggingFace token below first" : "Download"}
                        >
                          <Download className="w-4 h-4" />
                        </Button>
                      )}
                    </div>

                    {/* HuggingFace token row — only shown for gated models not yet downloaded */}
                    {needsToken && (
                      <div className="flex items-center gap-2 mt-1">
                        <Input
                          value={hfToken}
                          onChange={e => setHfToken(e.target.value)}
                          placeholder="hf_… (HuggingFace token — get one at hf.co/settings/tokens)"
                          className="h-7 text-xs font-mono"
                          type="password"
                        />
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}