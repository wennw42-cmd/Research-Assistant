import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { CompareReader } from './CompareReader';
import { samplePairs } from './sample-pairs';
import './styles.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <CompareReader pairs={samplePairs} title="Attention Is All You Need · 中英对照" />
  </StrictMode>,
);
