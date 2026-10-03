export type TrackingIdentityOverlaySource = 'fixed_roster' | 'round_state'
export type TrackingAssignmentStatus = 'assigned' | 'unassigned'

export interface TrackingIdentityObservation {
  frame_index: number
  timestamp_s: number
  x: number
  z: number
  bbox: [number, number, number, number] | null
  source_lens: string | null
  representative_view: string | null
  virtual_bbox: [number, number, number, number] | null
}

export interface TrackingIdentityTracklet {
  tracklet_id: string
  identity_id: string
  assignment_status: TrackingAssignmentStatus
  observations: TrackingIdentityObservation[]
}

export interface TrackingIdentityOverlayArtifact {
  source_id: string
  source_kind: TrackingIdentityOverlaySource
  state_id: string
  provisional: boolean
  identity_count: number
  assigned_tracklet_count: number
  unassigned_tracklet_count: number
  source_width: number
  source_height: number
  tracks: TrackingIdentityTracklet[]
}
