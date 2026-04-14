'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Skeleton } from '@/components/ui/skeleton'
import {
  BarChart3,
  Image as ImageIcon,
  Tag,
  Layers,
  TrendingUp,
  Database,
  RefreshCw,
  Download,
  ChevronRight,
  PenTool,
  ArrowUpDown,
  ArrowLeftRight,
  Cpu,
  PieChart,
  AlertCircle,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import type { Dataset } from '@/app/page'

interface DashboardViewProps {
  datasets: Dataset[]
  selectedDataset: Dataset | null
  onSelectDataset: (dataset: Dataset) => void
  setActiveView: (view: string) => void
  apiUrl: string
}

interface DatasetStats {
  dataset_id: string
  name: string
  format: string
  task_type: string
  total_images: number
  total_annotations: number
  classes: Record<string, any>
  class_distribution: Record<string, number>
  splits: Record<string, number>
  image_sizes: Record<string, any>
  avg_annotations_per_image: number
  created_at: string
}

const CLASS_COLORS = [
  'bg-blue-500', 'bg-emerald-500', 'bg-amber-500', 'bg-rose-500',
  'bg-violet-500', 'bg-cyan-500', 'bg-orange-500', 'bg-pink-500',
  'bg-teal-500', 'bg-indigo-500', 'bg-lime-500', 'bg-fuchsia-500',
]

const SPLIT_COLORS: Record<string, string> = {
  train: 'bg-primary',
  val: 'bg-emerald-500',
  valid: 'bg-emerald-500',
  validation: 'bg-emerald-500',
  test: 'bg-amber-500',
}

export function DashboardView({
  datasets,
  selectedDataset,
  onSelectDataset,
  setActiveView,
  apiUrl,
}: DashboardViewProps) {
  const [stats, setStats] = useState<DatasetStats | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [statsError, setStatsError] = useState<string | null>(null)

  useEffect(() => {
    if (selectedDataset) {
      setStats(null)
      setStatsError(null)
      loadStats()
    }
  }, [selectedDataset])

  const loadStats = async (forceRefresh = false) => {
    if (!selectedDataset) return
    setIsLoading(true)
    setStatsError(null)
    try {
      const url = `${apiUrl}/api/datasets/${selectedDataset.id}/stats${forceRefresh ? '?force_refresh=true' : ''}`
      const response = await fetch(url)
      if (response.ok) {
        const data = await response.json()
        setStats(data)
      } else {
        setStatsError('Failed to load statistics')
      }
    } catch {
      setStatsError('Could not reach backend')
    } finally {
      setIsLoading(false)
    }
  }

  const exportReport = () => {
    if (!stats || !selectedDataset) return
    const report = {
      dataset: {
        id: selectedDataset.id,
        name: selectedDataset.name,
        format: selectedDataset.format,
        task_type: selectedDataset.task_type,
        created_at: selectedDataset.created_at,
      },
      summary: {
        total_images: stats.total_images,
        total_annotations: stats.total_annotations,
        num_classes: Object.keys(stats.class_distribution || {}).length,
        avg_annotations_per_image: stats.avg_annotations_per_image,
      },
      class_distribution: stats.class_distribution,
      splits: stats.splits,
      classes: stats.classes,
      generated_at: new Date().toISOString(),
    }
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${selectedDataset.name}_report_${new Date().toISOString().split('T')[0]}.json`
    a.click()
    URL.revokeObjectURL(url)
    const csv = ['class,count,percentage']
    const total = Object.values(stats.class_distribution || {}).reduce((a: number, b: unknown) => a + (b as number), 0) as number
    Object.entries(stats.class_distribution || {}).forEach(([cls, count]) => {
      csv.push(`${cls},${count},${total > 0 ? (((count as number) / total) * 100).toFixed(2) : 0}%`)
    })
    const csvBlob = new Blob([csv.join('\n')], { type: 'text/csv' })
    const csvUrl = URL.createObjectURL(csvBlob)
    const csvA = document.createElement('a')
    csvA.href = csvUrl
    csvA.download = `${selectedDataset.name}_classes_${new Date().toISOString().split('T')[0]}.csv`
    setTimeout(() => { csvA.click(); URL.revokeObjectURL(csvUrl) }, 300)
  }

  const totalAnnotations = stats?.class_distribution
    ? Object.values(stats.class_distribution).reduce((a, b) => a + b, 0)
    : 0

  const sortedClasses = stats?.class_distribution
    ? Object.entries(stats.class_distribution).sort(([, a], [, b]) => b - a)
    : []

  /* ── No dataset selected ───────────────────────────────────────────── */
  if (!selectedDataset) {
    return (
      <div className="h-full flex flex-col p-6 overflow-y-auto">
        {/* Header */}
        <div className="mb-6">
          <p className="text-[10px] font-mono uppercase tracking-[0.2em] text-muted-foreground/50 mb-2">Dashboard</p>
          <h2 className="text-2xl font-display font-bold tracking-tight">Dataset Overview</h2>
          <p className="text-sm text-muted-foreground mt-1">
            Select a dataset to view its statistics and analytics.
          </p>
        </div>

        {datasets.length === 0 ? (
          /* No datasets at all — onboarding */
          <div className="flex-1 flex items-center justify-center">
            <div className="max-w-sm w-full text-center space-y-6">
              <div className="w-16 h-16 rounded-2xl bg-primary/8 border border-primary/15 flex items-center justify-center mx-auto">
                <Database className="w-7 h-7 text-primary" strokeWidth={1.5} />
              </div>

              <div className="space-y-2">
                <h3 className="text-lg font-display font-bold">No datasets loaded yet</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  Load an image dataset to start annotating, processing, and training AI models.
                </p>
              </div>

              {/* Quick-start steps */}
              <div className="text-left bg-muted/40 rounded-xl border border-border p-4 space-y-3">
                <p className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground/60">How it works</p>
                {[
                  { n: '1', label: 'Load a dataset', desc: 'Upload a ZIP or point to a local folder' },
                  { n: '2', label: 'Annotate images', desc: 'Draw boxes and labels on your images' },
                  { n: '3', label: 'Train an AI model', desc: 'Monitor loss and accuracy in real time' },
                ].map(({ n, label, desc }) => (
                  <div key={n} className="flex items-start gap-3">
                    <span className="w-5 h-5 rounded-full bg-primary/15 border border-primary/25 text-primary text-[9px] font-bold flex items-center justify-center shrink-0 mt-0.5">
                      {n}
                    </span>
                    <div>
                      <p className="text-xs font-medium">{label}</p>
                      <p className="text-[11px] text-muted-foreground">{desc}</p>
                    </div>
                  </div>
                ))}
              </div>

              <Button onClick={() => setActiveView('datasets')} className="w-full" size="lg">
                <Database className="w-4 h-4 mr-2" strokeWidth={1.75} />
                Load Your First Dataset
              </Button>
            </div>
          </div>
        ) : (
          /* Datasets exist — show picker */
          <div>
            <p className="text-xs text-muted-foreground mb-4">
              Click a dataset to view its statistics
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {datasets.map((dataset) => (
                <button
                  key={dataset.id}
                  className="p-4 rounded-xl border border-border bg-card hover:border-primary/30 hover:bg-primary/5 transition-all text-left group"
                  onClick={() => onSelectDataset(dataset)}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="w-8 h-8 rounded-lg bg-primary/8 border border-primary/15 flex items-center justify-center">
                      <Database className="w-3.5 h-3.5 text-primary" strokeWidth={1.75} />
                    </div>
                    <ChevronRight className="w-4 h-4 text-muted-foreground/30 group-hover:text-primary transition-colors" />
                  </div>
                  <p className="font-semibold text-sm truncate mb-1">{dataset.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {(!dataset.format || dataset.format === 'unknown' || dataset.format === 'generic-images') ? (dataset.format === 'generic-images' ? 'IMAGES' : 'UNKNOWN') : dataset.format.toUpperCase()} · {dataset.num_images.toLocaleString()} images · {dataset.classes?.length || 0} classes
                  </p>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    )
  }

  /* ── Dataset selected ──────────────────────────────────────────────── */
  return (
    <div className="p-6 space-y-5 overflow-y-auto h-full">

      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <p className="text-[10px] font-mono uppercase tracking-[0.2em] text-muted-foreground/50 mb-1">Dashboard</p>
          <h2 className="text-2xl font-display font-bold tracking-tight">{selectedDataset.name}</h2>
          <p className="text-sm text-muted-foreground mt-0.5">
            {(!selectedDataset.format || selectedDataset.format === 'unknown' ? 'Unknown format' : selectedDataset.format === 'generic-images' ? 'Images only' : selectedDataset.format.toUpperCase())} · {selectedDataset.task_type || 'unknown type'} · created {new Date(selectedDataset.created_at).toLocaleDateString()}
          </p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <Button variant="outline" size="sm" onClick={() => loadStats(true)} disabled={isLoading}>
            <RefreshCw className={cn('w-3.5 h-3.5 mr-1.5', isLoading && 'animate-spin')} strokeWidth={1.75} />
            Refresh
          </Button>
          <Button variant="outline" size="sm" onClick={exportReport} disabled={!stats}>
            <Download className="w-3.5 h-3.5 mr-1.5" strokeWidth={1.75} />
            Export Report
          </Button>
        </div>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {[
          {
            label: 'Total Images',
            value: (stats?.total_images ?? selectedDataset.num_images).toLocaleString(),
            icon: ImageIcon,
            iconColor: 'text-blue-500',
            iconBg: 'bg-blue-500/10',
            desc: 'Images in this dataset',
          },
          {
            label: 'Total Annotations',
            value: (stats?.total_annotations ?? selectedDataset.num_annotations).toLocaleString(),
            icon: Tag,
            iconColor: 'text-emerald-500',
            iconBg: 'bg-emerald-500/10',
            desc: 'Labels drawn on images',
          },
          {
            label: 'Classes',
            value: Object.keys(stats?.class_distribution || {}).length || selectedDataset.classes.length,
            icon: Layers,
            iconColor: 'text-amber-500',
            iconBg: 'bg-amber-500/10',
            desc: 'Distinct object categories',
          },
          {
            label: 'Avg per Image',
            value: (stats?.avg_annotations_per_image ?? (selectedDataset.num_images > 0 ? selectedDataset.num_annotations / selectedDataset.num_images : 0)).toFixed(1),
            icon: TrendingUp,
            iconColor: 'text-violet-500',
            iconBg: 'bg-violet-500/10',
            desc: 'Annotations per image',
          },
        ].map(({ label, value, icon: Icon, iconColor, iconBg, desc }) => (
          <div key={label} className="rounded-xl border border-border bg-card p-4">
            <div className="flex items-start justify-between mb-3">
              <p className="text-[10px] font-mono uppercase tracking-[0.12em] text-muted-foreground/70 leading-tight">{label}</p>
              <div className={cn('w-7 h-7 rounded-lg flex items-center justify-center shrink-0', iconBg)}>
                <Icon className={cn('w-3.5 h-3.5', iconColor)} strokeWidth={1.75} />
              </div>
            </div>
            <p className="text-3xl font-display font-bold tracking-tight">{value}</p>
            <p className="text-[10px] text-muted-foreground/60 mt-1">{desc}</p>
          </div>
        ))}
      </div>

      {/* Class distribution + Splits */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">

        {/* Class distribution */}
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-start justify-between">
              <div>
                <CardTitle className="text-sm font-semibold flex items-center gap-2">
                  <BarChart3 className="w-3.5 h-3.5 text-muted-foreground" strokeWidth={1.75} />
                  Class Distribution
                </CardTitle>
                <CardDescription className="text-[11px] mt-0.5">
                  How annotations are spread across object categories
                </CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 max-h-[340px] overflow-y-auto pr-1">
              {isLoading && !stats ? (
                /* Loading skeleton */
                Array.from({ length: 4 }).map((_, i) => (
                  <div key={i}>
                    <div className="flex items-center justify-between mb-1.5">
                      <Skeleton className="h-3 w-24 rounded" />
                      <Skeleton className="h-3 w-16 rounded" />
                    </div>
                    <Skeleton className="h-1.5 w-full rounded-full" />
                  </div>
                ))
              ) : statsError ? (
                <div className="flex flex-col items-center gap-2 py-6 text-center">
                  <AlertCircle className="w-4 h-4 text-destructive" />
                  <p className="text-xs text-muted-foreground">{statsError}</p>
                </div>
              ) : sortedClasses.length > 0 ? (
                sortedClasses.map(([className, count], idx) => {
                  const percentage = totalAnnotations > 0 ? (count / totalAnnotations) * 100 : 0
                  return (
                    <div key={className}>
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="text-xs font-medium truncate flex-1 mr-2">{className}</span>
                        <span className="text-xs text-muted-foreground tabular-nums shrink-0">
                          {count.toLocaleString()} <span className="text-muted-foreground/50">({percentage.toFixed(1)}%)</span>
                        </span>
                      </div>
                      <div className="h-1.5 bg-secondary rounded-full overflow-hidden">
                        <div
                          className={cn('h-full rounded-full transition-all', CLASS_COLORS[idx % CLASS_COLORS.length])}
                          style={{ width: `${percentage}%` }}
                        />
                      </div>
                    </div>
                  )
                })
              ) : (
                <p className="text-xs text-muted-foreground text-center py-6">
                  No class data available for this dataset
                </p>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Dataset splits */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-semibold flex items-center gap-2">
              <PieChart className="w-3.5 h-3.5 text-muted-foreground" strokeWidth={1.75} />
              Dataset Splits
            </CardTitle>
            <CardDescription className="text-[11px]">
              How images are divided between train, validation, and test sets
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading && !stats ? (
              /* Loading skeleton */
              <div className="space-y-4">
                {Array.from({ length: 3 }).map((_, i) => (
                  <div key={i}>
                    <div className="flex items-center justify-between mb-1.5">
                      <Skeleton className="h-3 w-12 rounded" />
                      <Skeleton className="h-3 w-20 rounded" />
                    </div>
                    <Skeleton className="h-2 w-full rounded-full" />
                  </div>
                ))}
              </div>
            ) : statsError ? (
              <div className="flex flex-col items-center gap-2 py-6 text-center">
                <AlertCircle className="w-4 h-4 text-destructive" />
                <p className="text-xs text-muted-foreground">{statsError}</p>
              </div>
            ) : stats?.splits && Object.keys(stats.splits).length > 0 ? (
              <div className="space-y-4">
                {Object.entries(stats.splits).map(([split, count]) => {
                  const totalImgs = stats?.total_images || selectedDataset.num_images || 1
                  const percentage = (count / totalImgs) * 100
                  const colorClass = SPLIT_COLORS[split] || 'bg-muted-foreground'
                  return (
                    <div key={split}>
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="text-xs font-medium capitalize">{split}</span>
                        <span className="text-xs text-muted-foreground tabular-nums">
                          {count.toLocaleString()} <span className="text-muted-foreground/50">({percentage.toFixed(1)}%)</span>
                        </span>
                      </div>
                      <div className="h-2 bg-secondary rounded-full overflow-hidden">
                        <div
                          className={cn('h-full rounded-full transition-all', colorClass)}
                          style={{ width: `${percentage}%` }}
                        />
                      </div>
                    </div>
                  )
                })}
              </div>
            ) : (
              <div className="text-center py-8 space-y-3">
                <p className="text-xs text-muted-foreground">
                  No splits configured. Split your dataset to separate training data from test data.
                </p>
                <Button variant="outline" size="sm" onClick={() => setActiveView('split')}>
                  Create Splits
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Dataset info */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold">Dataset Information</CardTitle>
          <CardDescription className="text-[11px]">Technical details about this dataset</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-5">
            {[
              { label: 'Format', value: (!selectedDataset.format || selectedDataset.format === 'unknown' ? 'Unknown' : selectedDataset.format === 'generic-images' ? 'Images (no annotations)' : selectedDataset.format.toUpperCase()) },
              { label: 'Task Type', value: (stats?.task_type || selectedDataset.task_type || 'Unknown') },
              { label: 'Created', value: new Date(selectedDataset.created_at).toLocaleDateString() },
              { label: 'Dataset ID', value: `${selectedDataset.id.slice(0, 12)}…`, mono: true },
            ].map(({ label, value, mono }) => (
              <div key={label}>
                <p className="text-[10px] font-mono uppercase tracking-[0.12em] text-muted-foreground/60 mb-1">{label}</p>
                <p className={cn('text-sm font-medium capitalize', mono && 'font-mono text-xs')}>{value}</p>
              </div>
            ))}
          </div>

          {selectedDataset.classes.length > 0 && (
            <div>
              <p className="text-[10px] font-mono uppercase tracking-[0.12em] text-muted-foreground/60 mb-2">Classes</p>
              <div className="flex flex-wrap gap-1.5">
                {selectedDataset.classes.map((cls, idx) => (
                  <span
                    key={cls}
                    className={cn(
                      'px-2 py-0.5 rounded-md text-[11px] font-medium text-white',
                      CLASS_COLORS[idx % CLASS_COLORS.length]
                    )}
                  >
                    {cls}
                  </span>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Quick actions */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold">Quick Actions</CardTitle>
          <CardDescription className="text-[11px]">Common next steps for this dataset</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              { view: 'annotate', icon: PenTool, label: 'Annotate', desc: 'Label images' },
              { view: 'sorting', icon: ArrowUpDown, label: 'Sort & Filter', desc: 'Organise data' },
              { view: 'convert', icon: ArrowLeftRight, label: 'Convert', desc: 'Change format' },
              { view: 'training', icon: Cpu, label: 'Train Model', desc: 'Start AI training' },
            ].map(({ view, icon: Icon, label, desc }) => (
              <button
                key={view}
                onClick={() => setActiveView(view)}
                className="h-auto py-4 flex flex-col items-center gap-1.5 rounded-xl border border-border bg-card hover:border-primary/30 hover:bg-primary/5 transition-all"
              >
                <Icon className="w-5 h-5 text-muted-foreground" strokeWidth={1.75} />
                <span className="text-xs font-semibold">{label}</span>
                <span className="text-[10px] text-muted-foreground/70">{desc}</span>
              </button>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
