export type MediaKind = 'image' | 'frame' | 'video' | 'clip' | 'resource'

export interface ViewContext {
  id: string; field_length: number; field_width: number; landmarks: Landmark[]
  uniforms: Record<string, { matrix?: number; value: number | number[] }>
  vertex_shader: string; fragment_shader: string
  tracking_geometry: { available: boolean; reason: string; logic2camera?: number[]; fps?: number; aspect?: number }
}

export interface Calibration {
  ownership?: 'managed' | 'external'
  version?: string | number | null
  id: string
  name: string
  path: string
  source: string
  field_length: number
  field_width: number
}

export interface MediaItem {
  ownership?: 'managed' | 'external'
  role?: 'tracking' | 'dual_lens' | 'camera_directions' | 'veo_highlights' | 'clip' | 'image' | 'unknown'
  size_bytes?: number
  id: string
  name: string
  path: string
  source: string
  kind: MediaKind
  width: number
  height: number
  fps: number | null
  duration_s: number | null
  frame_count?: number
  timestamp_s?: number
  extracted_from?: string
  url: string
}

export interface MatchWorkspace {
  import_status?: string | null
  metadata?: Record<string, string | null>
  id: string
  name: string
  description: string
  created_at: string
  updated_at: string
  calibration: Calibration | null
  media: MediaItem[]
  media_count?: number
  clips?: Clip[]
}

export interface ClipAnalysis {
  status: string
  detection_artifact_id?: string
  association_artifact_id?: string
  tracking_artifact_id?: string
  run_id?: string
}

export interface Clip {
  id: string
  source_media_id: string
  source_start_s?: number
  source_end_s?: number
  duration_s?: number
  start_s?: number
  end_s?: number
  clip_media_id?: string | null
  generation?: { status: string; job_id?: string; error?: string }
  name: string
  created_at: string
  analysis: ClipAnalysis | null
  debug_video_url?: string
}

export interface ClipJob {
  job_id: string
  match_id: string
  clip_id: string
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
  stage: 'generating' | 'processing' | 'detection' | 'tracking' | 'rendering'
  processed_frames: number
  total_frames: number
  progress: number
  error: string | null
  detection_artifact_id?: string | null
  tracking_artifact_id?: string | null
}

export interface Landmark { id: string; name: string; x: number; z: number }

export interface RenderedPanorama {
  timestamp_s?: number
  id: string
  url: string
  width: number
  height: number
  field_length: number
  field_width: number
  landmarks: Landmark[]
}

export interface GroundPoint {
  view?: string
  timestamp_s?: number
  media_id?: string
  view_width?: number
  pixel_x: number
  pixel_y: number
  u: number
  v: number
  world_x: number | null
  world_z: number | null
  distance: number | null
  inside_field: boolean | null
  status: 'inside_field' | 'outside_field' | 'no_ground_intersection' | 'camera_state_unavailable' | 'ray_parallel_to_ground' | 'invalid_ground_intersection'
  landmark: string | null
  landmark_name: string | null
  expected_x: number | null
  expected_z: number | null
  measured_x: number | null
  measured_z: number | null
  delta_x: number | null
  delta_z: number | null
  field_error_m: number | null
  point_index?: number
}

export interface VeoCandidate { calibration_version?: string | null; identifier: string; safe_url: string; filename: string; content_type: string | null; role: string; confidence: number; reasons: string[]; discovery_phases: string[]; width: number | null; height: number | null; fps: number | null; duration_s: number | null; field_length: number | null; field_width: number | null }
export interface VeoResourceReference { identifier: string; filename: string; safe_url: string; discovery_phases: string[] }
export interface VeoInspectionInteraction { default_phase_resources: VeoResourceReference[]; interactive_control_found: boolean; interactive_click_succeeded: boolean; interactive_activation_confirmed: boolean; activation_method?: string | null; primary_selector: string; primary_selector_found: boolean; primary_selector_count: number; primary_selector_visible: boolean; primary_selector_hover_required: boolean; primary_selector_click_succeeded: boolean; primary_selector_force_click_used: boolean; interactive_attempts: string[]; menu_controls_inspected: string[]; post_interactive_resources: VeoResourceReference[]; final_classified_assets: { tracking: VeoResourceReference[]; dual_lens: VeoResourceReference[]; calibration: VeoResourceReference[] } }
export interface VeoInspection { id: string; match_url: string; metadata: Record<string, string | null>; candidates: VeoCandidate[]; interaction: VeoInspectionInteraction; diagnostic_path: string | null; access_error?: string | null; resolved_assets: Record<string, string>; ready: boolean }
export interface VeoJob { progress?: number | null; assets: Record<string, { status: string; filename: string; bytes_downloaded: number; bytes_total: number | null }>; completed_assets?: number; identifier: string; status: string; current_asset: string | null; bytes_downloaded: number; bytes_total: number | null; error: string | null; match_id: string | null }
