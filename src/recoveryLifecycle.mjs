// Observation only: completion waits for both flow cleanup and the last
// SweetAlert view's destruction. Device readiness is a separate notification.
let nextRecoveryId = 0

export function createRecoveryLifecycle(notify) {
  const id = ++nextRecoveryId
  let view = 0
  let viewDestroyed = true
  let outcome = null
  let ended = false
  function emit(phase, result) {
    try {
      const returned = notify?.(Object.freeze({ id, phase, outcome: result }))
      if (returned) void Promise.resolve(returned).catch(() => {})
    } catch (_) {
      /* Observers cannot interrupt recovery. */
    }
  }
  function flush() {
    if (!ended && outcome && viewDestroyed) {
      ended = true
      emit('ended', outcome)
    }
  }
  emit('awaiting-resume')
  return {
    phase(phase) {
      if (!ended) emit(phase)
    },
    watchView() {
      const current = ++view
      viewDestroyed = false
      return () => {
        if (current !== view) return
        viewDestroyed = true
        flush()
      }
    },
    finish(result) {
      if (outcome) return
      outcome = result
      flush()
    },
  }
}
