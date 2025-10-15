import React from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { AutoMLProvider } from './context/automlcontext';
import App from './App';
import ReportPage from './components/reportpage';
import './index.css';

const container = document.getElementById('root');
const root = createRoot(container);

root.render(
  <React.StrictMode>
    <BrowserRouter>
      <AutoMLProvider>
        <Routes>
          <Route path="/*" element={<App />} />
          <Route path="/report" element={<ReportPage />} />
        </Routes>
      </AutoMLProvider>
    </BrowserRouter>
  </React.StrictMode>
);