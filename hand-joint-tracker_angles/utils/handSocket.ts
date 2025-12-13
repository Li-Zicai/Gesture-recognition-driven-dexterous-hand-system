// utils/handSocket.ts
export class HandSocket {
  private ws: WebSocket | null = null;
  private url: string;
  public onAck: ((data: any) => void) | null = null;

  constructor(url = "ws://localhost:8765") {
    this.url = url;
  }

  connect() {
    if (this.ws) this.ws.close();
    this.ws = new WebSocket(this.url);
    this.ws.onopen = () => console.log("HandSocket connected");
    this.ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (this.onAck) this.onAck(data);
      } catch (e) {
        console.warn("Invalid message", e);
      }
    };
    this.ws.onclose = () => {
      console.log("HandSocket closed, reconnecting in 1s");
      setTimeout(() => this.connect(), 1000);
    };
    this.ws.onerror = (e) => console.error("HandSocket error", e);
  }

  sendJoints(joints: number[]) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return false;
    const payload = { type: "joints", joints, ts: Date.now() };
    this.ws.send(JSON.stringify(payload));
    return true;
  }

  close() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}