// videoLiveMonitor.mjs
// emits a snapshot whenever the stream's "liveness" changes.
// status: "live" | "muted" | "inactive" | "ended"

const GRACE_PERIOD_MS = 5000;
const BAD_STATUS_THRESHOLD = 3;
const LOG_PREFIX = '[CameraMonitor]';

export class VideoLiveMonitor {
    constructor(stream, videoEl, pollMs = 1000) {
      this.stream = stream;
      this.video = videoEl;
      this.track = stream && stream.getVideoTracks()[0] || null;
      this.pollMs = pollMs;
  
      this._tickId = null;
      this._listeners = new Set();
      this._lastStatus = null;
      this._graceUntil = 0;
      this._consecutiveBadCount = 0;
      this._tickCount = 0;
      // > 0 while an external flow (e.g. RC's camera-selection UI) owns the
      // camera. While suspended the monitor neither polls nor listens, so
      // RC's own stream churn (previews, swaps, teardown) can't read as a
      // participant-side disconnect. Pairs with suspend()/unsuspend().
      this._suspended = 0;
  
      // bind handlers
      //
      // `track.ended` is a definitive signal that the active video
      // device is gone (unplugged, OS revoked permission, OS-level
      // device error). It's NOT the kind of transient bad status the
      // grace period + 3-strike threshold is meant to filter out — yet
      // `_emitIfChanged()` would still suppress it during the 5s
      // post-switch grace window. That suppression breaks downstream
      // flows that legitimately need to react to the camera vanishing
      // shortly after the participant chose it (e.g. the unknown-camera
      // confirmation modal). Use `_emitImmediate` so a real `ended`
      // event always fires, regardless of grace / threshold state.
      this._onEnded = () => { console.warn(LOG_PREFIX, 'Track "ended" event fired — emitting immediately (bypass grace/threshold)'); this._emitImmediate('ended'); };
      this._onMuteUnmute = () => { console.log(LOG_PREFIX, 'Track mute/unmute event fired, muted:', this.track?.muted); this._emitIfChanged(); };
      this._onDeviceChange = () => { console.log(LOG_PREFIX, 'devicechange event fired'); this._emitIfChanged(); };

      console.log(LOG_PREFIX, 'Constructed. Track:', this.track?.readyState, 'Stream active:', stream?.active);
    }
  
    onChange(fn) {
      this._listeners.add(fn);
      const snap = this._snapshot();
      console.log(LOG_PREFIX, 'onChange registered, initial snapshot:', snap.status, snap);
      fn(snap);
    }
    offChange(fn) { this._listeners.delete(fn); }
  
    start() {
      if (!this.track) {
        console.warn(LOG_PREFIX, 'start() called but no track available');
        return;
      }
      if (this._suspended > 0) {
        console.log(LOG_PREFIX, `start() skipped — suspended (depth ${this._suspended})`);
        return;
      }
      console.log(LOG_PREFIX, 'start() — attaching listeners, pollMs:', this.pollMs, 'graceUntil:', this._graceUntil > 0 ? `${Math.round((this._graceUntil - Date.now()) / 1000)}s remaining` : 'none');
  
      this.track.addEventListener("ended", this._onEnded, { once: false });
      this.track.addEventListener("mute", this._onMuteUnmute, { once: false });
      this.track.addEventListener("unmute", this._onMuteUnmute, { once: false });
  
      if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
        navigator.mediaDevices.addEventListener("devicechange", this._onDeviceChange);
      }
  
      this._tickCount = 0;
      const tick = () => {
        this._tickCount++;
        this._emitIfChanged();
        this._tickId = setTimeout(tick, this.pollMs);
      };
      this._tickId = setTimeout(tick, this.pollMs);
    }
  
    pause() {
      console.log(LOG_PREFIX, 'pause() — stopping polling and detaching listeners');
      if (this._tickId) { clearTimeout(this._tickId); this._tickId = null; }
      this._detachTrackListeners();
    }

    /**
     * Suspend disconnect detection while another flow owns the camera
     * (RC's Choose Camera / Choose Screen popups open and close their own
     * preview streams and swap cameras). Nestable; unsuspend() re-arms on
     * the current stream with a fresh grace period once the depth hits 0.
     */
    suspend() {
      this._suspended++;
      console.log(LOG_PREFIX, `suspend() — depth ${this._suspended}`);
      this.pause();
    }

    unsuspend() {
      if (this._suspended > 0) this._suspended--;
      console.log(LOG_PREFIX, `unsuspend() — depth ${this._suspended}`);
      if (this._suspended === 0 && this.stream && this.track) {
        this.updateStream(this.stream, this.video);
      }
    }

    stop() {
      console.log(LOG_PREFIX, 'stop() — full stop, clearing listeners');
      this.pause();
      this._listeners.clear();
    }

    updateStream(newStream, videoEl) {
      console.log(LOG_PREFIX, 'updateStream() — new stream active:', newStream?.active, 'new track readyState:', newStream?.getVideoTracks()[0]?.readyState);
      if (this._tickId) { clearTimeout(this._tickId); this._tickId = null; }
      this._detachTrackListeners();

      this.stream = newStream;
      if (videoEl) this.video = videoEl;
      this.track = newStream && newStream.getVideoTracks()[0] || null;
      this._lastStatus = null;
      this._consecutiveBadCount = 0;
      this._graceUntil = Date.now() + GRACE_PERIOD_MS;

      console.log(LOG_PREFIX, `updateStream() — grace period set for ${GRACE_PERIOD_MS}ms, starting monitor`);
      this.start();
    }

    _detachTrackListeners() {
      if (this.track) {
        this.track.removeEventListener("ended", this._onEnded);
        this.track.removeEventListener("mute", this._onMuteUnmute);
        this.track.removeEventListener("unmute", this._onMuteUnmute);
      }
      if (navigator.mediaDevices && navigator.mediaDevices.removeEventListener) {
        navigator.mediaDevices.removeEventListener("devicechange", this._onDeviceChange);
      }
    }
  
    _snapshot() {
      const track = this.track;
      const streamActive = !!(this.stream && this.stream.active);
      const trackReady = track ? track.readyState : "none";
      const muted = !!(track && track.muted);
      const vrs = this.video ? this.video.readyState : 0;
  
      let status;
      if (!streamActive) status = "inactive";
      else if (!track || trackReady === "ended") status = "ended";
      else if (muted) status = "muted";
      else status = "live";
  
      return {
        status, trackReadyState: trackReady, muted, streamActive, videoReadyState: vrs
      };
    }

    // Force an immediate emission with the given status, bypassing both
    // the post-switch grace window and the consecutive-bad-status
    // threshold. Reserved for events whose meaning is unambiguous
    // (e.g. MediaStreamTrack's `ended` event), so that legitimate
    // disconnects are never silently swallowed.
    _emitImmediate(forcedStatus) {
      const baseSnap = this._snapshot();
      const snap = forcedStatus ? { ...baseSnap, status: forcedStatus } : baseSnap;
      console.warn(LOG_PREFIX, `_emitImmediate() — forcing status "${snap.status}" (was "${this._lastStatus}"), notifying ${this._listeners.size} listener(s)`);
      this._lastStatus = snap.status;
      this._consecutiveBadCount = 0;
      this._graceUntil = 0;
      this._listeners.forEach(fn => fn(snap));
    }

    _emitIfChanged() {
      const now = Date.now();
      if (now < this._graceUntil) {
        if (this._tickCount <= 2) {
          console.log(LOG_PREFIX, `_emitIfChanged() — in grace period (${Math.round((this._graceUntil - now) / 1000)}s left), skipping`);
        }
        return;
      }

      const snap = this._snapshot();
      const isBad = snap.status === 'ended' || snap.status === 'inactive';

      if (isBad) {
        this._consecutiveBadCount++;
        console.warn(LOG_PREFIX, `_emitIfChanged() — BAD status "${snap.status}", consecutiveBadCount: ${this._consecutiveBadCount}/${BAD_STATUS_THRESHOLD}`, snap);
        if (this._consecutiveBadCount < BAD_STATUS_THRESHOLD) return;
        console.error(LOG_PREFIX, `_emitIfChanged() — threshold reached! Firing disconnect. status: ${snap.status}`);
      } else {
        if (this._consecutiveBadCount > 0) {
          console.log(LOG_PREFIX, `_emitIfChanged() — status recovered to "${snap.status}", resetting badCount from ${this._consecutiveBadCount}`);
        }
        this._consecutiveBadCount = 0;
      }

      if (snap.status !== this._lastStatus) {
        console.log(LOG_PREFIX, `_emitIfChanged() — status changed: "${this._lastStatus}" → "${snap.status}", notifying ${this._listeners.size} listener(s)`);
        this._lastStatus = snap.status;
        this._listeners.forEach(fn => fn(snap));
      }
    }
  }
