import { Landmark, TARGET_JOINTS_CONFIG } from '../types';

export const drawHand = (
  ctx: CanvasRenderingContext2D,
  landmarks: Landmark[],
  width: number,
  height: number
) => {
  ctx.clearRect(0, 0, width, height);

  const connections = [
    [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8], // Index
    [0, 9], [9, 10], [10, 11], [11, 12], // Middle
    [0, 13], [13, 14], [14, 15], [15, 16], // Ring
    [0, 17], [17, 18], [18, 19], [19, 20] // Pinky
  ];

  // Bones
  ctx.lineWidth = 2;
  ctx.strokeStyle = '#334155'; // slate-700
  
  for (const [start, end] of connections) {
    const p1 = landmarks[start];
    const p2 = landmarks[end];
    ctx.beginPath();
    ctx.moveTo(p1.x * width, p1.y * height);
    ctx.lineTo(p2.x * width, p2.y * height);
    ctx.stroke();
  }

  // Identify tracked indices
  const trackedIndices = new Set<number>(TARGET_JOINTS_CONFIG.map(j => j.index));
  
  // Note: Thumb needs specific care because we track 1,2,3 for 4 values.
  // Actually we track indices 1, 2, 3. Tip (4) is used for calculation but is not a "joint" pivot per se in our mapping?
  // User said "Dip, Pip, Mcp, Abd".
  // Our mapping: Abd(at 1), Mcp(at 1?), Pip(at 2), Dip(at 3).
  // We should highlight 1, 2, 3. 0 is Wrist. 
  
  landmarks.forEach((landmark, index) => {
    const x = landmark.x * width;
    const y = landmark.y * height;

    ctx.beginPath();
    
    if (trackedIndices.has(index)) {
        // Tracked Joint - Green
        ctx.fillStyle = '#10b981'; 
        ctx.arc(x, y, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = '#064e3b';
        ctx.lineWidth = 1;
        ctx.stroke();
    } else {
        // Untracked (Tips mostly) - Small Gray
        ctx.fillStyle = '#64748b'; 
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.fill();
    }
  });
};