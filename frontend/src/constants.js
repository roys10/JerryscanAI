// Manufacturing-line camera order and the UI fallback when the backend is not
// ready to publish a model-specific angle contract.
export const ANGLES = [
  { id: 'G01', label: 'G01' },
  { id: 'G02', label: 'G02' },
  { id: 'G03', label: 'G03' },
  { id: 'G04', label: 'G04' },
];

export const ANGLE_IDS = ANGLES.map(angle => angle.id);
