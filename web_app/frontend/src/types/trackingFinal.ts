export type FinalPositionMode = 'smoothed' | 'measurement'
export type PointOrigin = 'observed' | 'fused' | 'interpolated' | string

export interface FinalPoint {
  identity_id: string
  segment_id: string
  frame_index: number
  timestamp_s: number
  measurement_x: number
  measurement_z: number
  smoothed_x: number
  smoothed_z: number
  origin: PointOrigin
  speed_mps: number | null
  acceleration_mps2: number | null
  position_adjustment_m: number | null
  position_std_m: number | null
  kinematics_quality: string | null
}

export interface FinalTrajectory { identity_id: string; points: FinalPoint[]; [key: string]: unknown }
export interface SegmentBreak { identity_id: string; left_frame_index: number; right_frame_index: number; [key: string]: unknown }
export interface TrackingFinalArtifact { index: Record<string, unknown>; summary: Record<string, unknown>; trajectories: FinalTrajectory[]; segment_breaks: SegmentBreak[] }
export interface PersistentPoint {
  actor_id: string; segment_id: string; frame_index: number; timestamp_s: number
  x: number; z: number; measurement_x: number; measurement_z: number; origin: PointOrigin
  speed_mps?: number | null; position_std_m?: number | null; quality?: string | null
}
export interface PersistentTrajectory { actor_id: string; points: PersistentPoint[]; [key: string]: unknown }
export interface PersistentGap { actor_id: string; start_frame_index: number; end_frame_index: number; [key: string]: unknown }
export interface TrackingPersistentArtifact { index: Record<string, unknown>; summary: Record<string, unknown>; audit: Record<string, unknown> | null; trajectories: PersistentTrajectory[]; gaps: PersistentGap[] }
export type TrackingFieldPoint = FinalPoint | PersistentPoint
export type TrackingFieldTrajectory = FinalTrajectory | PersistentTrajectory
export interface TrackingReview { identity_id?: string; frame_index?: number; start_frame_index?: number; timestamp_s?: number; start_timestamp_s?: number; segment_id?: string; [key: string]: unknown }
export interface TrackingReviews { speed_reviews: TrackingReview[]; acceleration_reviews: TrackingReview[]; adjustment_reviews: TrackingReview[]; uncertainty_reviews: TrackingReview[]; short_segments: TrackingReview[]; [key: string]: unknown }
