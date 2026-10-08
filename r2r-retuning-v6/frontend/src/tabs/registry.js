import {
  BarChart3, Calculator, Factory, Gauge, GitCompare, LineChart,
  RefreshCw, Sigma, Sliders, Waves,
} from 'lucide-react';

// Two top-level entries. Twin Study is the dashboard's main character; the ten
// paper-reproduction views live one level down behind Validation, reached by
// the sub-selector in App.jsx rather than by their own nav entries.
export const PAGE_GROUPS = [
  {
    id: 'main',
    label: 'Dashboard',
    pages: [
      { id: 'twinstudy', label: 'Twin Study', icon: Sliders },
      { id: 'validation', label: 'Validation', icon: GitCompare },
    ],
  },
];

export const PAGES = PAGE_GROUPS.flatMap((group) => group.pages);

// The ten existing views, unchanged in id, label and order. App.jsx keys its
// existing ternary chain on these ids via `validationPage`, so none of the
// page components themselves are touched.
export const VALIDATION_PAGES = [
  { id: 'simulation', label: 'Simulation', icon: Waves },
  { id: 'plants', label: 'Plants', icon: Factory },
  { id: 'sysid', label: 'SysID', icon: Sigma },
  { id: 'logging', label: 'Logging rate', icon: LineChart },
  { id: 'excitation', label: 'Excitation', icon: BarChart3 },
  { id: 'noiseLpf', label: 'Noise-aware logging (LPF)', icon: LineChart },
  { id: 'damping', label: 'Closed-loop damping', icon: Gauge },
  { id: 'retuning', label: 'Retuning (§4)', icon: RefreshCw },
  { id: 'equations', label: 'Equation', icon: Calculator },
  { id: 'drift', label: 'Drift', icon: GitCompare },
];
