'use client'

import { useState, useEffect, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { Slider } from '@/components/ui/slider'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import {
  Copy, Trash2, Loader2, AlertCircle, CheckCircle2,
  ScanSearch, LayoutGrid, Cpu, Hash
} from 'lucide-react'
import { cn } from '@/lib/utils'
import type { Dataset } from '@/app/page'
import { toast } from 'sonner'

interface DuplicateDetectionViewProps {
  selectedDataset: Dataset | null
  apiUrl: string
}

interface DuplicateItem {
  path: string
  full_path: string
  hash?: string
  distance?: number
  similarity?: number
  is_original: boolean
}

interface ScanResult {
  success: boolean
  method: string
  threshold: number
  total_images: number
  duplicate_groups: number
  total_duplicates: number
  unique_images: number
  groups: DuplicateItem[][]
}

export function DuplicateDetectionView({ selectedDataset, apiUrl }: DuplicateDetectionViewProps) {
  const [method, setMethod] = useState<'perceptual' | 'average' | 'md5' | 'clip'>('perceptual')
  const [threshold, setThreshold] = useState(10)
  const [keepStrategy, setKeepStrategy] = useState<'first' | 'largest' | 'smallest'>('first')
  const [isScanning, setIsScanning] = useState(false)
  const [isRemoving, setIsRemoving] = useState(false)
  const [result, setResult] = useState<ScanResult | null>(null)
  // track which item in each group is selected to keep (index within group)
  const [keepSelections, setKeepSelections] = useState<Record<number, number>>({})
  // CLIP embedding cache tracking
  const [clipEmbeddingsReady, setClipEmbeddingsReady] = useState(false)
  const [clipEmbeddingsDatasetId, setClipEmbeddingsDatasetId] = useState<string | null>(null)
  // AbortController for the current scan fetch
  const scanAbortRef = useRef<AbortController | null>(null)
  // Track which dataset_id has an active CLIP scan (for backend cancel)
  const activeScanDatasetRef = useRef<string | null>(null)

  const cancelActiveScan = (datasetId: string | null) => {
    if (!datasetId) return
    // Abort the in-flight fetch
    scanAbortRef.current?.abort()
    // Tell the backend to stop GPU computation
    navigator.sendBeacon(`${apiUrl}/api/datasets/${datasetId}/cancel-scan`)
    activeScanDatasetRef.current = null
  }

  // Cancel scan when switching datasets
  useEffect(() => {
    cancelActiveScan(activeScanDatasetRef.current)
    setResult(null)
    setKeepSelections({})
    setClipEmbeddingsReady(false)
    setClipEmbeddingsDatasetId(null)
  }, [selectedDataset?.id])

  // Cancel scan when component unmounts (navigating away)
  useEffect(() => {
    return () => {
      cancelActiveScan(activeScanDatasetRef.current)
    }
  }, [])

  // Cancel scan on page reload / tab close
  useEffect(() => {
    const handleUnload = () => cancelActiveScan(activeScanDatasetRef.current)
    window.addEventListener('beforeunload', handleUnload)
    return () => window.removeEventListener('beforeunload', handleUnload)
  }, [])

  const scan = async () => {
    if (!selectedDataset) return
    setIsScanning(true)
    setResult(null)
    setKeepSelections({})
    const abortController = new AbortController()
    scanAbortRef.current = abortController
    if (method === 'clip') activeScanDatasetRef.current = selectedDataset.id
    try {
      const resp = await fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/find-duplicates`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          method,
          threshold: method === 'clip' ? Math.round(threshold) : threshold,
          include_near_duplicates: true,
        }),
        signal: abortController.signal,
      })
      if (!resp.ok) throw new Error('Scan failed')
      const data: ScanResult = await resp.json()
      if (!data.success) throw new Error((data as any).error ?? 'Scan failed')
      setResult(data)
      // Default: keep first item in each group
      const defaults: Record<number, number> = {}
      data.groups.forEach((_, gi) => { defaults[gi] = 0 })
      setKeepSelections(defaults)
      if (method === 'clip') {
        setClipEmbeddingsReady(true)
        setClipEmbeddingsDatasetId(selectedDataset.id)
      }
      if (data.duplicate_groups === 0) {
        toast.success('No duplicates found — dataset looks clean!')
      } else {
        toast.info(`Found ${data.duplicate_groups} duplicate group(s) with ${data.total_duplicates} removable image(s)`)
      }
    } catch (err) {
      if ((err as any)?.name !== 'AbortError') {
        toast.error(err instanceof Error ? err.message : 'Scan failed')
      }
    } finally {
      activeScanDatasetRef.current = null
      scanAbortRef.current = null
      setIsScanning(false)
    }
  }

  const regroup = async () => {
    if (!selectedDataset || !clipEmbeddingsReady) return
    setIsScanning(true)
    try {
      const resp = await fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/clip-regroup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ threshold: Math.round(threshold) }),
      })
      if (!resp.ok) throw new Error('Regroup failed')
      const data: ScanResult = await resp.json()
      if (!data.success) throw new Error((data as any).error ?? 'Regroup failed')
      setResult(data)
      const defaults: Record<number, number> = {}
      data.groups.forEach((_, gi) => { defaults[gi] = 0 })
      setKeepSelections(defaults)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Regroup failed')
    } finally {
      setIsScanning(false)
    }
  }

  const removeSelected = async () => {
    if (!selectedDataset || !result) return
    // Build groups with the user's keep selection applied
    const groups = result.groups.map((group, gi) => {
      const keepIdx = keepSelections[gi] ?? 0
      // Rotate so the kept item is first
      return [...group.slice(keepIdx), ...group.slice(0, keepIdx)]
    })
    setIsRemoving(true)
    try {
      const resp = await fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/remove-duplicates`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ groups, keep_strategy: 'first' }),
      })
      if (!resp.ok) throw new Error('Remove failed')
      const data = await resp.json()
      toast.success(`Removed ${data.removed_count} duplicate image(s)`)
      setResult(null)
      setKeepSelections({})
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Remove failed')
    } finally {
      setIsRemoving(false)
    }
  }

  const totalToRemove = result
    ? result.groups.reduce((sum, group) => sum + group.length - 1, 0)
    : 0

  if (!selectedDataset) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <AlertCircle className="w-12 h-12 text-muted-foreground/50 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-muted-foreground">No dataset selected</h3>
          <p className="text-sm text-muted-foreground/70 mt-1">Select a dataset from the Datasets view</p>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Header + controls */}
      <div className="shrink-0 border-b border-border bg-card px-6 py-4 space-y-4">
        <div className="flex items-center gap-3">
          <Copy className="w-5 h-5 text-primary" strokeWidth={1.5} />
          <div>
            <h2 className="font-display font-bold text-sm tracking-tight">Duplicate Detection</h2>
            <p className="text-[10px] text-muted-foreground font-mono">
              {selectedDataset.name} · {selectedDataset.num_images?.toLocaleString()} images
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-end gap-4">
          {/* Method */}
          <div className="space-y-1">
            <label className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground/60">Method</label>
            <Select value={method} onValueChange={v => setMethod(v as typeof method)}>
              <SelectTrigger className="h-8 w-48 text-xs bg-background border-border">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="perceptual">
                  <span className="flex items-center gap-2"><Hash className="w-3 h-3" /> Perceptual Hash (pHash)</span>
                </SelectItem>
                <SelectItem value="average">
                  <span className="flex items-center gap-2"><Hash className="w-3 h-3" /> Average Hash (aHash)</span>
                </SelectItem>
                <SelectItem value="md5">
                  <span className="flex items-center gap-2"><Hash className="w-3 h-3" /> MD5 (exact only)</span>
                </SelectItem>
                <SelectItem value="clip">
                  <span className="flex items-center gap-2"><Cpu className="w-3 h-3" /> CLIP Embeddings (semantic)</span>
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Threshold */}
          {method !== 'md5' && (
            <div className="space-y-1 w-52">
              <div className="flex items-center justify-between">
                <label className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground/60">
                  {method === 'clip' ? 'Similarity %' : 'Hash Distance'}
                </label>
                <span className="text-xs font-mono text-primary">
                  {method === 'clip' ? `${threshold}%` : threshold}
                </span>
              </div>
              <Slider
                value={[threshold]}
                onValueChange={([v]) => setThreshold(v)}
                min={method === 'clip' ? 80 : 0}
                max={method === 'clip' ? 99 : 64}
                step={1}
              />
              <div className="flex items-center justify-between gap-2">
                <p className="text-[9px] text-muted-foreground/50">
                  {method === 'clip'
                    ? 'Higher % = only near-identical images'
                    : 'Lower = stricter (0 = exact match)'}
                </p>
                {method === 'clip' && clipEmbeddingsReady && clipEmbeddingsDatasetId === selectedDataset?.id && (
                  <button
                    onClick={regroup}
                    disabled={isScanning || isRemoving}
                    className="shrink-0 text-[9px] font-mono px-1.5 py-0.5 rounded bg-primary/10 text-primary hover:bg-primary/20 disabled:opacity-40 transition-colors"
                    title="Re-apply threshold using cached embeddings"
                  >
                    Re-apply
                  </button>
                )}
              </div>
            </div>
          )}

          {/* Keep strategy (only shown when results exist) */}
          {result && result.duplicate_groups > 0 && (
            <div className="space-y-1">
              <label className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground/60">Keep Strategy</label>
              <Select value={keepStrategy} onValueChange={v => setKeepStrategy(v as typeof keepStrategy)}>
                <SelectTrigger className="h-8 w-36 text-xs bg-background border-border">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="first">Keep First</SelectItem>
                  <SelectItem value="largest">Keep Largest</SelectItem>
                  <SelectItem value="smallest">Keep Smallest</SelectItem>
                </SelectContent>
              </Select>
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center gap-2 ml-auto">
            {method === 'clip' && clipEmbeddingsReady && clipEmbeddingsDatasetId === selectedDataset?.id && (
              <span className="text-[9px] font-mono text-success/80 flex items-center gap-1">
                <Cpu className="w-2.5 h-2.5" /> embeddings cached
              </span>
            )}
            <Button size="sm" onClick={scan} disabled={isScanning || isRemoving} className="gap-1.5">
              {isScanning
                ? <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Scanning…</>
                : <><ScanSearch className="w-3.5 h-3.5" strokeWidth={1.5} /> Scan Dataset</>
              }
            </Button>
            {result && result.duplicate_groups > 0 && (
              <Button
                size="sm"
                variant="destructive"
                onClick={removeSelected}
                disabled={isRemoving || isScanning}
                className="gap-1.5"
              >
                {isRemoving
                  ? <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Removing…</>
                  : <><Trash2 className="w-3.5 h-3.5" strokeWidth={1.5} /> Remove {totalToRemove} Duplicate{totalToRemove !== 1 ? 's' : ''}</>
                }
              </Button>
            )}
          </div>
        </div>

        {/* Stats bar */}
        {result && (
          <div className="flex flex-wrap items-center gap-4 pt-1 border-t border-border/60">
            {[
              { label: 'Total Images', value: result.total_images, color: '' },
              { label: 'Duplicate Groups', value: result.duplicate_groups, color: result.duplicate_groups > 0 ? 'text-warning' : 'text-success' },
              { label: 'Removable', value: result.total_duplicates, color: result.total_duplicates > 0 ? 'text-destructive' : 'text-muted-foreground/40' },
              { label: 'Unique', value: result.unique_images, color: 'text-success' },
            ].map(({ label, value, color }) => (
              <div key={label} className="flex items-baseline gap-1.5">
                <span className={cn('text-lg font-mono font-bold tabular-nums', color || 'text-foreground')}>{value}</span>
                <span className="text-[9px] font-mono text-muted-foreground/50 uppercase tracking-widest">{label}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Results */}
      <div className="flex-1 overflow-y-auto p-6">
        {/* Empty / idle state */}
        {!result && !isScanning && (
          <div className="flex flex-col items-center justify-center h-full gap-4 text-center">
            <LayoutGrid className="w-12 h-12 text-muted-foreground/30" strokeWidth={1} />
            <div>
              <p className="text-sm font-medium text-muted-foreground">Run a scan to find duplicates</p>
              <p className="text-xs text-muted-foreground/60 mt-1">
                Perceptual hash finds visually similar images. CLIP finds semantically similar images.
              </p>
              <p className="text-xs text-muted-foreground/50 mt-0.5">
                Combined mode: run Perceptual + CLIP for best coverage.
              </p>
            </div>
          </div>
        )}

        {/* Loading state */}
        {isScanning && (
          <div className="flex flex-col items-center justify-center h-full gap-4">
            <Loader2 className="w-10 h-10 animate-spin text-primary" />
            <p className="text-sm font-medium">Scanning {selectedDataset.num_images?.toLocaleString()} images…</p>
            {method === 'clip' && (
              <p className="text-xs text-muted-foreground">CLIP model loading may take a moment on first run</p>
            )}
          </div>
        )}

        {/* No duplicates */}
        {result && result.duplicate_groups === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-4 text-center">
            <CheckCircle2 className="w-12 h-12 text-success" strokeWidth={1.5} />
            <div>
              <p className="text-sm font-medium text-success">No duplicates found</p>
              <p className="text-xs text-muted-foreground/60 mt-1">
                All {result.total_images.toLocaleString()} images are unique at the current threshold.
              </p>
            </div>
          </div>
        )}

        {/* Duplicate groups */}
        {result && result.groups.length > 0 && (
          <div className="space-y-6">
            <p className="text-[10px] font-mono text-muted-foreground/50 uppercase tracking-widest">
              {result.groups.length} group{result.groups.length !== 1 ? 's' : ''} — click an image to mark it as the one to keep
            </p>
            {result.groups.map((group, gi) => {
              const keepIdx = keepSelections[gi] ?? 0
              return (
                <div key={gi} className="border border-border rounded-xl overflow-hidden">
                  {/* Group header */}
                  <div className="flex items-center gap-3 px-4 py-2.5 bg-card border-b border-border">
                    <span className="text-[9px] font-mono text-muted-foreground/40 uppercase tracking-widest">
                      Group {gi + 1}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {group.length} similar images
                    </span>
                    {group[1]?.distance !== undefined && (
                      <span className="text-[10px] font-mono text-muted-foreground/60 ml-auto">
                        hash dist: {group[1].distance}
                      </span>
                    )}
                    {group[1]?.similarity !== undefined && (
                      <span className="text-[10px] font-mono text-muted-foreground/60 ml-auto">
                        similarity: {(group[1].similarity * 100).toFixed(1)}%
                      </span>
                    )}
                  </div>

                  {/* Images grid */}
                  <div className="p-3 flex flex-wrap gap-3 bg-background/50">
                    {group.map((item, ii) => {
                      const isKept = ii === keepIdx
                      // path is relative to dataset root e.g. "images/train/file.jpg"
                      const imgSrc = `${apiUrl}/api/datasets/${selectedDataset.id}/image-file/${item.path}`
                      return (
                        <button
                          key={ii}
                          onClick={() => setKeepSelections(prev => ({ ...prev, [gi]: ii }))}
                          className={cn(
                            'relative rounded-lg overflow-hidden border-2 transition-all duration-150 shrink-0',
                            'focus:outline-none focus-visible:ring-2 focus-visible:ring-primary',
                            isKept
                              ? 'border-success shadow-lg shadow-success/10'
                              : 'border-border hover:border-destructive/40 opacity-70 hover:opacity-100'
                          )}
                          title={isKept ? 'Keeping this image' : 'Click to keep this image instead'}
                        >
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={imgSrc}
                            alt={item.path}
                            className="w-32 h-32 object-cover block"
                            onError={e => { (e.target as HTMLImageElement).style.display = 'none' }}
                          />
                          {/* Keep/delete badge */}
                          <div className={cn(
                            'absolute top-1.5 right-1.5 flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[9px] font-mono font-semibold',
                            isKept
                              ? 'bg-success text-white'
                              : 'bg-destructive/80 text-white'
                          )}>
                            {isKept ? <CheckCircle2 className="w-2.5 h-2.5" /> : <Trash2 className="w-2.5 h-2.5" />}
                            {isKept ? 'Keep' : 'Delete'}
                          </div>
                          {/* Filename tooltip */}
                          <div className="absolute bottom-0 inset-x-0 bg-black/60 px-1.5 py-1">
                            <p className="text-[9px] text-white/90 font-mono truncate">
                              {item.path.split('/').pop()}
                            </p>
                          </div>
                        </button>
                      )
                    })}
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
