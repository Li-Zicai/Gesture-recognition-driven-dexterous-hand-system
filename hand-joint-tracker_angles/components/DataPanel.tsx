import React from 'react';
import { JointData } from '../types';

interface DataPanelProps {
  data: JointData[];
}

const DataPanel: React.FC<DataPanelProps> = ({ data }) => {
  return (
    <div className="w-full bg-slate-800 rounded-xl p-4 border border-slate-700 shadow-lg h-[650px] flex flex-col">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          Joint Data
        </h2>
        <span className={`text-xs px-2 py-1 rounded font-mono transition-colors ${data.length > 0 ? 'bg-emerald-900 text-emerald-400' : 'bg-amber-900 text-amber-500'}`}>
            {data.length > 0 ? 'DETECTED' : 'SEARCHING'}
        </span>
      </div>

      <div className="flex-1 overflow-y-auto no-scrollbar pr-1 relative">
        {data.length === 0 ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 gap-6 p-6 text-center border-2 border-dashed border-slate-700 rounded-xl bg-slate-800/50">
            <div className="relative">
                <div className="w-16 h-16 rounded-full border-4 border-slate-600"></div>
                <div className="absolute top-0 left-0 w-16 h-16 rounded-full border-t-4 border-emerald-500 animate-spin"></div>
                <svg className="w-8 h-8 text-slate-400 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 11.5V14m0-2.5v-6a1.5 1.5 0 113 0m-3 6a1.5 1.5 0 00-3 0v2a7.5 7.5 0 0015 0v-5a1.5 1.5 0 00-3 0m-6-3V11m0-5.5v-1a1.5 1.5 0 013 0v1m0 0V11m0-5.5a1.5 1.5 0 013 0v3m0 0V11" />
                </svg>
            </div>
            <div>
                <h4 className="text-white font-medium mb-1">Waiting for Hand</h4>
                <p className="text-xs text-slate-400 leading-relaxed">
                    Please raise your hand clearly in front of the camera to view joint data.
                </p>
            </div>
            <div className="grid grid-cols-2 gap-2 w-full">
                 <div className="h-1.5 bg-slate-700 rounded-full animate-pulse w-full"></div>
                 <div className="h-1.5 bg-slate-700 rounded-full animate-pulse w-2/3"></div>
                 <div className="h-1.5 bg-slate-700 rounded-full animate-pulse w-3/4"></div>
                 <div className="h-1.5 bg-slate-700 rounded-full animate-pulse w-full"></div>
            </div>
          </div>
        ) : (
          <div className="space-y-4 pb-2">
             {/* Wrist */}
             <GroupSection title="Base" data={data.filter(d => d.group === 'Wrist')} />
             {/* Thumb */}
             <GroupSection title="Thumb (4 DOF)" data={data.filter(d => d.group === 'Thumb')} />
             {/* Fingers */}
             <GroupSection title="Index" data={data.filter(d => d.group === 'Index')} />
             <GroupSection title="Middle" data={data.filter(d => d.group === 'Middle')} />
             <GroupSection title="Ring" data={data.filter(d => d.group === 'Ring')} />
             <GroupSection title="Pinky" data={data.filter(d => d.group === 'Pinky')} />
          </div>
        )}
      </div>
    </div>
  );
};

const GroupSection: React.FC<{ title: string, data: JointData[] }> = React.memo(({ title, data }) => (
    <div className="bg-slate-700/30 rounded-lg p-3 border border-slate-700/50">
        <h4 className="text-[10px] uppercase font-bold text-slate-400 mb-2 tracking-wider flex justify-between">
            {title}
        </h4>
        <div className="space-y-2">
            {data.map((joint) => (
                <JointRow key={joint.name} joint={joint} />
            ))}
        </div>
    </div>
));

const JointRow: React.FC<{ joint: JointData }> = React.memo(({ joint }) => {
    // Determine Max Range for visualization
    // Abduction is small range (0-30), Flexion is large (0-110)
    const maxVal = joint.type === 'Abduction' ? 30 : 120; 
    const percentage = Math.min(100, Math.max(0, (joint.angle / maxVal) * 100));

    <span className={`font-mono text-xs ${joint.visibility && joint.visibility < 0.5 ? 'text-red-400' : 'text-slate-400'}`}>
      {joint.visibility ? `Conf: ${(joint.visibility*100).toFixed(0)}%` : ''}
    </span>
    
    // Determine Color based on type and value
    let barColor = '#10b981'; // emerald-500 default
    if (joint.type === 'Abduction') {
        barColor = '#3b82f6'; // blue-500 for abduction
    } else {
        // Flexion Gradient
        if (joint.angle > 90) barColor = '#ef4444'; // red
        else if (joint.angle > 45) barColor = '#f59e0b'; // amber
    }

    return (
        <div className="flex flex-col gap-1.5">
            <div className="flex justify-between items-center text-xs">
                <span className="text-slate-300 font-medium w-24 flex items-center gap-1.5">
                    {joint.type === 'Abduction' && <span className="w-1 h-1 rounded-full bg-blue-500"></span>}
                    {joint.name.split(' ')[1]}
                </span>
                <span className="font-mono text-white opacity-90 tabular-nums">{joint.angle.toFixed(0)}°</span>
            </div>
            <div className="w-full bg-slate-900 h-1.5 rounded-full overflow-hidden flex items-center">
                <div 
                    style={{ 
                        width: `${percentage}%`,
                        backgroundColor: barColor,
                    }}
                    className="h-full rounded-full"
                ></div>
            </div>
        </div>
    );
});

export default React.memo(DataPanel);