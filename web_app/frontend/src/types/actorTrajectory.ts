export type ActorTrajectoryOrigin = 'observed' | 'interpolated'
export type TrajectoryBbox = [number, number, number, number]

export interface ActorTrajectoryVisualEndpoint {
  detection_id: string | null
  frame_index: number
  timestamp_s: number
  representative_view: string
  virtual_bbox_xyxy: TrajectoryBbox
  source_lens: string | null
  source_bbox_xyxy: TrajectoryBbox | null
}

export interface ActorTrajectoryVisualInterpolation {
  left: ActorTrajectoryVisualEndpoint
  right: ActorTrajectoryVisualEndpoint
  alpha: number
  visual_interpolation_allowed: boolean
  source_bbox_interpolation_allowed: boolean
}

export interface ActorTrajectoryPoint {
  frame_index: number
  timestamp_s: number
  x: number
  z: number
  vx_mps: number | null
  vz_mps: number | null
  speed_mps: number | null
  origin: ActorTrajectoryOrigin
  segment_id: string | null
  segment_index: number | null
  source_tracklet_ids: string[]
  visual_anchor: ActorTrajectoryVisualEndpoint | null
  visual_interpolation: ActorTrajectoryVisualInterpolation | null
}

export interface ActorTrajectory {
  actor_id: string
  tracklet_ids: string[]
  segment_count: number
  points: ActorTrajectoryPoint[]
}

export interface ActorTrajectoryArtifact {
  trajectory_version: 'actor-trajectory-reconstruction-v1'
  provisional: true
  actor_count: number
  display_point_count: number
  observed_point_count: number
  interpolated_point_count: number
  segment_count: number
  estimated_fps: number | null
  frame_step: number | null
  time_step_s: number | null
  actors: ActorTrajectory[]
}
