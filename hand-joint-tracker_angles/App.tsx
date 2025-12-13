import React, { useState, useEffect, useRef } from 'react';
import HandTracker from './components/HandTracker';
import DataPanel from './components/DataPanel';
import { HandSocket } from './utils/handSocket'; // 改动1
import { JointData } from './types';

const App: React.FC = () => {
  const [jointData, setJointData] = useState<JointData[]>([]);
  // 在组件内部（在 useState 声明之后）
  const socketRef = useRef<HandSocket | null>(null);
  const lastSentRef = useRef<number>(0);
  const SEND_INTERVAL_MS = 50; // 20Hz，可改成 33(30Hz) / 100(10Hz) 等

  useEffect(() => {
    socketRef.current = new HandSocket("ws://localhost:8765"); // 本地测试地址
    socketRef.current.connect();
    socketRef.current.onAck = (d) => console.log("hand ack:", d);

    return () => {
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, []);

  // 发送 jointData（节流 + 映射）
  useEffect(() => {
    if (!socketRef.current) return;
    if (!jointData || jointData.length === 0) return;

    const now = Date.now();
    if (now - lastSentRef.current < SEND_INTERVAL_MS) return;
    lastSentRef.current = now;

    // 映射：取 angle 字段。根据你的硬件调整转换函数
    let joints = jointData.map(j => j.angle); // 示例：度数数组
    // 确保长度为 17（必要时填充/裁剪）
    if (joints.length > 17) joints = joints.slice(0, 17);
    if (joints.length < 17) {
      const pad = Array(17 - joints.length).fill(0);
      joints = joints.concat(pad);
    }

    // 可选：阈值过滤（仅在变化显著时发送）
    // 例如：只发送当某个关节 > 1 度变化等（需要保存上次值）
    socketRef.current.sendJoints(joints);
  }, [jointData]);

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col">
      {/* Header */}
      <header className="bg-slate-800 border-b border-slate-700 py-4 shadow-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 flex justify-between items-center">
          <div className="flex items-center gap-3">
             <div className="p-2 bg-emerald-500/20 rounded-lg">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 11.5V14m0-2.5v-6a1.5 1.5 0 113 0m-3 6a1.5 1.5 0 00-3 0v2a7.5 7.5 0 0015 0v-5a1.5 1.5 0 00-3 0m-6-3V11m0-5.5v-1a1.5 1.5 0 013 0v1m0 0V11m0-5.5a1.5 1.5 0 013 0v3m0 0V11" />
                </svg>
             </div>
             <h1 className="text-xl font-bold tracking-tight">Hand Joint AI</h1>
          </div>
          <div className="text-sm text-slate-400 hidden sm:block">
            Powered by MediaPipe & React
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-grow container mx-auto px-4 py-8">
        <div className="flex flex-col lg:flex-row gap-6">
            
            {/* Left Column: Visualizer */}
            <div className="lg:w-2/3 flex flex-col gap-4">
               <div className="bg-slate-800 rounded-xl p-1 border border-slate-700 shadow-lg">
                  <div className="bg-black rounded-lg overflow-hidden relative">
                     <HandTracker onResults={setJointData} />
                  </div>
               </div>
               
               <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700/50">
                  <h3 className="text-lg font-semibold text-white mb-2">Instructions</h3>
                  <ul className="list-disc list-inside text-slate-400 text-sm space-y-1">
                      <li>Grant camera permissions when prompted.</li>
                      <li>Position your hand clearly in front of the camera.</li>
                      <li>The AI detects 21 points but filters for the <strong>17 specific joints</strong> requested.</li>
                      <li><span className="text-emerald-400 font-medium">Green points</span> indicate the targeted 17 joints.</li>
                      <li><span className="text-slate-500 font-medium">Gray points</span> indicate fingertips (not in data list).</li>
                  </ul>
               </div>
            </div>

            {/* Right Column: Data */}
            <div className="lg:w-1/3">
                <DataPanel data={jointData} />
            </div>

        </div>
      </main>

      {/* Footer */}
      <footer className="bg-slate-800 border-t border-slate-700 py-6 mt-auto">
        <div className="container mx-auto px-4 text-center text-slate-500 text-sm">
          <p>&copy; {new Date().getFullYear()} Hand Joint AI. Client-side Computer Vision.</p>
        </div>
      </footer>
    </div>
  );
};

export default App;
