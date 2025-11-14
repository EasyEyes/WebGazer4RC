// videoLiveMonitor.mjs
// emits a snapshot whenever the stream's "liveness" changes.
// status: "live" | "muted" | "inactive" | "ended"

export class VideoLiveMonitor {
    constructor(stream, videoEl, pollMs = 1000) {
      this.stream = stream;
      this.video = videoEl;
      this.track = stream && stream.getVideoTracks()[0] || null;
      this.pollMs = pollMs;
  
      this._tickId = null;
      this._listeners = new Set();
      this._lastStatus = null;
  
      // bind handlers
      this._onEnded = this._emitIfChanged.bind(this);
      this._onMuteUnmute = this._emitIfChanged.bind(this);
      this._onDeviceChange = this._emitIfChanged.bind(this);
    }
  
    onChange(fn) {
      this._listeners.add(fn);
      fn(this._snapshot()); // fire immediately with current state
    }
    offChange(fn) { this._listeners.delete(fn); }
  
    start() {
      if (!this.track) return;
  
      this.track.addEventListener("ended", this._onEnded, { once: false });
      this.track.addEventListener("mute", this._onMuteUnmute, { once: false });
      this.track.addEventListener("unmute", this._onMuteUnmute, { once: false });
  
      if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
        navigator.mediaDevices.addEventListener("devicechange", this._onDeviceChange);
      }
  
      const tick = () => {
        this._emitIfChanged();
        this._tickId = setTimeout(tick, this.pollMs);
      };
      this._tickId = setTimeout(tick, this.pollMs);
    }
  
    stop() {
      if (this._tickId) { clearTimeout(this._tickId); this._tickId = null; }
      if (this.track) {
        this.track.removeEventListener("ended", this._onEnded);
        this.track.removeEventListener("mute", this._onMuteUnmute);
        this.track.removeEventListener("unmute", this._onMuteUnmute);
      }
      if (navigator.mediaDevices && navigator.mediaDevices.removeEventListener) {
        navigator.mediaDevices.removeEventListener("devicechange", this._onDeviceChange);
      }
      this._listeners.clear();
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
  
    _emitIfChanged() {
      const snap = this._snapshot();
      if (snap.status !== this._lastStatus) {
        this._lastStatus = snap.status;
        this._listeners.forEach(fn => fn(snap));
      }
    }
  }
  