import React, { useEffect, useRef, useState, useCallback } from 'react';
import { FilesetResolver, HandLandmarker, Landmark } from '@mediapipe/tasks-vision';
import { drawHand } from '../utils/drawingUtils';
import { TARGET_JOINTS_CONFIG, JointData } from '../types';

interface HandTrackerProps {
  onResults: (data: JointData[]) => void;
}

// --- 1. One Euro Filter (抗抖动平滑算法) ---
class OneEuroFilter {
  minCutoff: number;
  beta: number;
  dCutoff: number;
  xPrev: number | null = null;
  dxPrev: number | null = null;
  tPrev: number | null = null;

  constructor(minCutoff = 1.0, beta = 0.0, dCutoff = 1.0) {
    this.minCutoff = minCutoff; 
    this.beta = beta;           
    this.dCutoff = dCutoff;
  }

  filter(val: number, timestamp: number = performance.now()): number {
    if (this.xPrev === null || this.tPrev === null || this.dxPrev === null) {
      this.xPrev = val;
      this.dxPrev = 0;
      this.tPrev = timestamp;
      return val;
    }

    const dt = (timestamp - this.tPrev) / 1000;
    if (dt === 0) return this.xPrev;

    this.tPrev = timestamp;
    const dx = (val - this.xPrev) / dt;
    const edx = this.lowPassFilter(dx, this.dxPrev, dt, this.dCutoff);
    this.dxPrev = edx;
    const cutoff = this.minCutoff + this.beta * Math.abs(edx);
    const result = this.lowPassFilter(val, this.xPrev, dt, cutoff);
    this.xPrev = result;
    return result;
  }

  lowPassFilter(x: number, xPrev: number, dt: number, cutoff: number) {
    const rc = 1.0 / (2.0 * Math.PI * cutoff);
    const alpha = dt / (dt + rc);
    return xPrev + alpha * (x - xPrev);
  }

  reset() {
    this.xPrev = null;
    this.dxPrev = null;
    this.tPrev = null;
  }
}

const HandTracker: React.FC<HandTrackerProps> = ({ onResults }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [landmarker, setLandmarker] = useState<HandLandmarker | null>(null);
  const [webcamRunning, setWebcamRunning] = useState(false);
  const [loading, setLoading] = useState(true);
  const [isMirrored, setIsMirrored] = useState(true);
  
  // 滤波器存储
  const filtersRef = useRef<Map<string, OneEuroFilter>>(new Map());
  
  const requestRef = useRef<number>(0);
  const webcamRunningRef = useRef(false);
  const landmarkerRef = useRef<HandLandmarker | null>(null);

  useEffect(() => {
    webcamRunningRef.current = webcamRunning;
    if (!webcamRunning) {
        if (requestRef.current) cancelAnimationFrame(requestRef.current);
        onResults([]); 
        filtersRef.current.forEach(f => f.reset());
    }
  }, [webcamRunning, onResults]);

  useEffect(() => {
    landmarkerRef.current = landmarker;
  }, [landmarker]);

  useEffect(() => {
    const createLandmarker = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
        );
        const handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 1,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5
        });
        setLandmarker(handLandmarker);
        setLoading(false);
      } catch (error) {
        console.error("Error loading MediaPipe Hand Landmarker:", error);
        setLoading(false);
      }
    };
    createLandmarker();

    return () => {
        if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, []);

  // --- 2. Math Helpers ---

  const getSmoothedValue = (key: string, newValue: number) => {
      if (!filtersRef.current.has(key)) {
          // minCutoff=1.0 (静止时稳), beta=0.007 (运动时跟手)
          filtersRef.current.set(key, new OneEuroFilter(1.0, 0.007)); 
      }
      return filtersRef.current.get(key)!.filter(newValue);
  };

  const toVector = (a: Landmark, b: Landmark) => ({
      x: a.x - b.x,
      y: a.y - b.y,
      z: a.z - b.z
  });

  const normalize = (v: {x: number, y: number, z: number}) => {
      const len = Math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
      return len === 0 ? {x:0, y:0, z:0} : {x: v.x/len, y: v.y/len, z: v.z/len};
  };

  const dotProduct = (v1: {x: number, y: number, z: number}, v2: {x: number, y: number, z: number}) => 
      v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;

  const calculateAngle = (v1: {x: number, y: number, z: number}, v2: {x: number, y: number, z: number}) => {
      const dot = dotProduct(normalize(v1), normalize(v2));
      const clamped = Math.max(-1, Math.min(1, dot));
      return (Math.acos(clamped) * 180) / Math.PI;
  };

  // 使用余弦定理计算三点夹角 
  const calculate3PointAngle = (p1: Landmark, p2: Landmark, p3: Landmark) => {
      const a = Math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2);
      const b = Math.sqrt((p3.x-p2.x)**2 + (p3.y-p2.y)**2 + (p3.z-p2.z)**2);
      const c = Math.sqrt((p1.x-p3.x)**2 + (p1.y-p3.y)**2 + (p1.z-p3.z)**2);
      if (a * b === 0) return 0;
      let cosine = (a**2 + b**2 - c**2) / (2 * a * b);
      cosine = Math.max(-1, Math.min(1, cosine));
      const angle = (Math.acos(cosine) * 180) / Math.PI;
      return Math.max(0, 180 - angle);
  };

  const calculateFingerAbduction = (mcp: Landmark, pip: Landmark, wrist: Landmark, middleMCP: Landmark) => {
      // 相对 手掌中心轴 (Wrist -> MiddleMCP) 计算外展
      const fingerVec = toVector(pip, mcp);
      const handAxis = toVector(middleMCP, wrist);
      return calculateAngle(fingerVec, handAxis);
  };

  const calculateThumbAbduction = (wrist: Landmark, thumbCMC: Landmark, indexMCP: Landmark) => {
      const vThumb = toVector(thumbCMC, wrist);
      const vIndex = toVector(indexMCP, wrist);
      return calculateAngle(vThumb, vIndex);
  };

  // 校准函数：解决伸直不归零(offset)和弯曲范围小(gain)的问题
  const remapAngle = (val: number, offset: number, gain: number) => {
      return Math.max(0, (val - offset) * gain);
  };

  // --- Main Loop ---

  const predictWebcam = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const currentLandmarker = landmarkerRef.current;

    if (!currentLandmarker || !video || !canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (video.videoWidth > 0 && video.videoHeight > 0) {
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        }
    }

    let startTimeMs = performance.now();
    if (video.currentTime > 0 && video.videoWidth > 0) {
       const results = currentLandmarker.detectForVideo(video, startTimeMs);
       ctx.clearRect(0, 0, canvas.width, canvas.height);
       
       if (results.landmarks && results.landmarks.length > 0) {
         const handLandmarks = results.landmarks[0];
         
         ctx.save();
         if (isMirrored) {
             ctx.translate(canvas.width, 0);
             ctx.scale(-1, 1);
         }
         drawHand(ctx, handLandmarks, canvas.width, canvas.height);
         ctx.restore();

         // 优先使用 worldLandmarks (3D 米单位) 进行物理计算
         const lm = results.worldLandmarks && results.worldLandmarks.length > 0 
            ? results.worldLandmarks[0] 
            : handLandmarks;

         const computedData: JointData[] = TARGET_JOINTS_CONFIG.map(config => {
            let angle = 0;

            if (config.name === 'Wrist Pitch') {
                const handDir = toVector(lm[9], lm[0]);
                const upVec = { x: 0, y: -1, z: 0 }; 
                angle = calculateAngle(handDir, upVec);
            }
            else if (config.group === 'Thumb') {
                if (config.name === 'Thumb ABD') angle = calculateThumbAbduction(lm[0], lm[1], lm[5]);
                else if (config.name === 'Thumb MCP') angle = calculate3PointAngle(lm[1], lm[2], lm[3]);
                else if (config.name === 'Thumb PIP') angle = calculate3PointAngle(lm[2], lm[3], lm[4]);
                else if (config.name === 'Thumb DIP') {
                    // [大拇指 DIP] 复用 IP 关节数据，确保 UI 响应
                    angle = calculate3PointAngle(lm[2], lm[3], lm[4]);
                }
            }
            else {
                // Fingers
                const mcpIdx = config.index;
                const pipIdx = config.index + 1;
                const dipIdx = config.index + 2;

                if (config.type === 'Abduction') {
                    angle = calculateFingerAbduction(lm[mcpIdx], lm[pipIdx], lm[0], lm[9]);
                } 
                else if (config.type === 'Flexion') {
                    if (config.name.includes('MCP')) {
                        angle = calculate3PointAngle(lm[0], lm[mcpIdx], lm[pipIdx]);
                        // MCP 通常比较准，但也可能有 5-10 度的自然偏差
                        angle = remapAngle(angle, 5, 1.0); 
                    }
                    else if (config.name.includes('PIP')) {
                        angle = calculate3PointAngle(lm[mcpIdx], lm[pipIdx], lm[dipIdx]);
                        
                        if (config.name === 'Index PIP') {
                            // [食指 PIP 强力校准]
                            // 减去 35 度(消除伸直时的45度幻影)
                            // 放大 2.5 倍(将弯曲后的剩余度数拉伸到正常范围)
                            angle = remapAngle(angle, 35, 2.5);
                        } else {
                            // 其他手指 PIP 轻微校准
                            angle = remapAngle(angle, 10, 1.1);
                        }
                    }
                    // 注意：此处不处理其他手指的 DIP，直接保持 angle=0 或跳过
                }
            }

            return {
                ...config,
                angle: getSmoothedValue(config.name, angle)
            };
         });
         
         onResults(computedData);
       } else {
         onResults([]); 
       }
    }

    if (webcamRunningRef.current) {
      requestRef.current = requestAnimationFrame(predictWebcam);
    }
  }, [onResults, isMirrored]);

  const toggleCam = async () => {
    if (!landmarker) return;
    
    if (webcamRunning) {
      setWebcamRunning(false);
      if (videoRef.current && videoRef.current.srcObject) {
         const stream = videoRef.current.srcObject as MediaStream;
         stream.getTracks().forEach(t => t.stop());
         videoRef.current.srcObject = null;
      }
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { 
              width: 1280, 
              height: 720,
              frameRate: { ideal: 60 } 
          } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.addEventListener('loadeddata', () => {
             webcamRunningRef.current = true;
             predictWebcam();
        });
        setWebcamRunning(true);
      }
    } catch (err) {
      alert("Camera access denied.");
    }
  };

  return (
    <div className="relative w-full max-w-3xl mx-auto rounded-2xl overflow-hidden shadow-2xl border border-slate-700 bg-black aspect-video group">
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-900 z-50">
             <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-500"></div>
        </div>
      )}

      {!webcamRunning && !loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-900/90 z-40 flex-col gap-4">
           <h3 className="text-white text-xl font-bold">Hardware Joint Tracker</h3>
           <p className="text-slate-400 text-sm">Calibrated for Index PIP & Thumb DIP</p>
           <button onClick={toggleCam} className="px-8 py-3 bg-emerald-600 hover:bg-emerald-700 text-white font-bold rounded-lg shadow-lg transition-colors">
             Start Camera
           </button>
        </div>
      )}

      <video ref={videoRef} className={`absolute inset-0 w-full h-full object-cover ${isMirrored ? 'scale-x-[-1]' : ''}`} autoPlay playsInline muted />
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full object-cover pointer-events-none" />

      {webcamRunning && (
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex gap-3 z-50 bg-slate-900/80 backdrop-blur px-4 py-2 rounded-full border border-slate-700 shadow-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300">
              <button 
                onClick={toggleCam}
                className="flex items-center gap-2 px-4 py-1.5 bg-red-500/20 hover:bg-red-500/40 text-red-400 rounded-full text-sm font-medium transition-colors"
              >
                  <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
                  Stop
              </button>
              
              <div className="w-px bg-slate-700 mx-1"></div>

              <button 
                onClick={() => setIsMirrored(m => !m)}
                className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${isMirrored ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-700/50 text-slate-400'}`}
              >
                  Mirror {isMirrored ? 'On' : 'Off'}
              </button>
          </div>
      )}
    </div>
  );
};

export default React.memo(HandTracker);