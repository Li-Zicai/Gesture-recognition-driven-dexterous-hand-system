export interface Landmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

export interface JointData {
  name: string;
  group: string; // 'Thumb', 'Index', etc.
  type: 'Flexion' | 'Abduction' | 'Base';
  index: number; // Main MediaPipe landmark index associated
  angle: number; // Degree value
}

export const TARGET_JOINTS_CONFIG = [
  // 1. Wrist
  { name: 'Wrist Pitch', group: 'Wrist', type: 'Base', index: 0 },

  // 2. Thumb (4 Joints: ABD, MCP, PIP, DIP)
  // Mapping MediaPipe: CMC(1), MCP(2), IP(3), TIP(4)
  { name: 'Thumb ABD', group: 'Thumb', type: 'Abduction', index: 1 }, // CMC Abduction
  { name: 'Thumb MCP', group: 'Thumb', type: 'Flexion', index: 1 },   // CMC Flexion (using index 1 as pivot)
  { name: 'Thumb PIP', group: 'Thumb', type: 'Flexion', index: 2 },   // MCP Flexion
  { name: 'Thumb DIP', group: 'Thumb', type: 'Flexion', index: 3 },   // IP Flexion

  // 3. Index (3 Joints: ABD, MCP, PIP)
  { name: 'Index ABD', group: 'Index', type: 'Abduction', index: 5 },
  { name: 'Index MCP', group: 'Index', type: 'Flexion', index: 5 },
  { name: 'Index PIP', group: 'Index', type: 'Flexion', index: 6 },

  // 4. Middle (3 Joints: ABD, MCP, PIP)
  { name: 'Middle ABD', group: 'Middle', type: 'Abduction', index: 9 },
  { name: 'Middle MCP', group: 'Middle', type: 'Flexion', index: 9 },
  { name: 'Middle PIP', group: 'Middle', type: 'Flexion', index: 10 },

  // 5. Ring (3 Joints: ABD, MCP, PIP)
  { name: 'Ring ABD', group: 'Ring', type: 'Abduction', index: 13 },
  { name: 'Ring MCP', group: 'Ring', type: 'Flexion', index: 13 },
  { name: 'Ring PIP', group: 'Ring', type: 'Flexion', index: 14 },

  // 6. Pinky (3 Joints: ABD, MCP, PIP)
  { name: 'Pinky ABD', group: 'Pinky', type: 'Abduction', index: 17 },
  { name: 'Pinky MCP', group: 'Pinky', type: 'Flexion', index: 17 },
  { name: 'Pinky PIP', group: 'Pinky', type: 'Flexion', index: 18 },
] as const;
