export interface TrackingIdentityObservation {
  timestamp_s: number
  bbox: [number, number, number, number]
  source_lens: string | null
}

export interface TrackingIdentityTracklet {
  tracklet_id: string
  identity_id: string
  observations: TrackingIdentityObservation[]
}

export interface TrackingIdentityOverlayArtifact {
  state_id: string
  provisional: boolean
  identity_count: number
  source_width: number
  source_height: number
  tracks: TrackingIdentityTracklet[]
}
