export interface VideoFrameMetadata { mediaTime: number }

export interface PresentationVideo {
  currentTime: number
  requestVideoFrameCallback?: (callback: (now: number, metadata: VideoFrameMetadata) => void) => number
  cancelVideoFrameCallback?: (handle: number) => void
}

interface ClockEnvironment {
  requestAnimationFrame: (callback: FrameRequestCallback) => number
  cancelAnimationFrame: (handle: number) => void
}

const browserEnvironment = (): ClockEnvironment => ({
  requestAnimationFrame: window.requestAnimationFrame.bind(window),
  cancelAnimationFrame: window.cancelAnimationFrame.bind(window),
})

/** Drive visual state from the media element that is actually being presented. */
export function startVideoPresentationClock(video: PresentationVideo, onFrame: (mediaTime: number) => void, environment = browserEnvironment()) {
  let videoFrameHandle: number | undefined
  let animationHandle: number | undefined
  let cancelled = false
  let previousTime = Number.NaN
  const publish = (mediaTime: number) => {
    if (Number.isFinite(mediaTime) && mediaTime !== previousTime) {
      previousTime = mediaTime
      onFrame(mediaTime)
    }
  }
  if (typeof video.requestVideoFrameCallback === 'function') {
    const callback = (_now: number, metadata: VideoFrameMetadata) => {
      if (cancelled) return
      publish(metadata.mediaTime)
      videoFrameHandle = video.requestVideoFrameCallback!(callback)
    }
    videoFrameHandle = video.requestVideoFrameCallback(callback)
    return () => {
      cancelled = true
      if (videoFrameHandle !== undefined) video.cancelVideoFrameCallback?.(videoFrameHandle)
    }
  }
  const tick = () => {
    if (cancelled) return
    publish(video.currentTime)
    animationHandle = environment.requestAnimationFrame(tick)
  }
  animationHandle = environment.requestAnimationFrame(tick)
  return () => {
    cancelled = true
    if (animationHandle !== undefined) environment.cancelAnimationFrame(animationHandle)
  }
}
