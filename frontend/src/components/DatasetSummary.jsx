// src/components/DatasetSummary.jsx
import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import {
  BarChart3,
  Target,
  AlertTriangle,
  Database,
  Hash,
} from 'lucide-react';
import Plot from 'react-plotly.js';
import axios from 'axios';

/* ----------  small UI pieces  ---------- */
const DualToggle = ({ view, setView }) => (
  <div className="flex gap-2">
    {['bar', 'lollipop'].map((v) => (
      <button
        key={v}
        onClick={() => setView(v)}
        className={`px-3 py-1 rounded-md text-sm capitalize ${
          view === v
            ? 'bg-teal-600 text-white'
            : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
        } transition`}
      >
        {v}
      </button>
    ))}
  </div>
);

const Top3Cards = ({ data }) => {
  const top3 = React.useMemo(
    () =>
      data.columns
        .map((c, i) => ({ col: c, gain: data.gains[i] }))
        .sort((a, b) => b.gain - a.gain)
        .slice(0, 3),
    [data]
  );

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
      {top3.map(({ col, gain }, idx) => (
        <div
          key={col}
          className="bg-gradient-to-br from-teal-50 to-cyan-50 border border-teal-200 rounded-lg p-4"
        >
          <div className="text-2xl font-bold text-teal-700">#{idx + 1}</div>
          <div className="text-sm text-gray-700 mt-1 truncate">{col}</div>
          <div className="text-xl font-semibold text-teal-900">
            {gain.toFixed(3)} bits
          </div>
        </div>
      ))}
    </div>
  );
};

const EntropyHeatMap = ({ data }) => {
  const top5 = React.useMemo(
    () =>
      data.columns
        .map((c, i) => ({ col: c, gain: data.gains[i] }))
        .sort((a, b) => b.gain - a.gain)
        .slice(0, 5),
    [data]
  );

  const heat = {
    z: top5.map(({ gain }) => [1.0 /*H(D)*/, 1.0 - gain /*H(D|A)*/]),
    x: ['H(D)', 'H(D|A)'],
    y: top5.map((x) => x.col),
    type: 'heatmap',
    colorscale: 'Blues',
    showscale: false,
  };

  return (
    <div className="mt-4">
      <Plot
        data={[heat]}
        layout={{
          title: 'Top-5: Entropy Before vs After Split',
          xaxis: { title: null },
          yaxis: { automargin: true },
          height: 220,
          font: { family: 'Inter, system-ui, sans-serif' },
        }}
        config={{ displayModeBar: false, responsive: true }}
      />
    </div>
  );
};

/* ----------  main component  ---------- */
const DatasetSummary = ({ uploadInfo }) => {
  /* ---- guard: wait for data ---- */
  if (!uploadInfo) {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 text-gray-500 text-center">
        Waiting for dataset…
      </div>
    );
  }

  const { stats, numerical_data_for_plot } = uploadInfo;
  const [entropyData, setEntropyData] = useState(null);
  const [view, setView] = useState('lollipop'); // 'bar' | 'lollipop'

  /* ---- fetch entropy gain ---- */
  useEffect(() => {
    if (!uploadInfo?.upload_path) return;
    axios
      .post('http://localhost:8000/entropy_gain', {
        upload_path: uploadInfo.upload_path,
        target_column: stats.target,
      })
      .then((res) => setEntropyData(res.data))
      .catch(() => {});
  }, [uploadInfo, stats.target]);

  /* ---- summary cards (original) ---- */
  const summaryCards = [
    {
      title: 'Dataset Shape',
      value: `${stats.shape[0]} rows, ${stats.shape[1]} columns`,
      icon: Database,
      color: 'blue',
    },
    {
      title: 'Target Column',
      value: stats.target || 'Not specified',
      icon: Target,
      color: 'green',
    },
    {
      title: 'Missing Values',
      value: `${Object.values(stats.n_missing).reduce((a, b) => a + b, 0)} total`,
      icon: AlertTriangle,
      color: 'amber',
    },
    {
      title: 'Data Types',
      value: `${Object.keys(stats.dtypes).length} columns`,
      icon: BarChart3,
      color: 'purple',
    },
  ];

  const colorClasses = {
    blue: 'bg-blue-50 text-blue-700 border-blue-200',
    green: 'bg-green-50 text-green-700 border-green-200',
    amber: 'bg-amber-50 text-amber-700 border-amber-200',
    purple: 'bg-purple-50 text-purple-700 border-purple-200',
  };

  /* ---- color buckets ---- */
  const gainColor = (g) => {
    if (g >= 0.3) return '#059669'; // emerald
    if (g >= 0.1) return '#f59e0b'; // amber
    return '#dc2626'; // red
  };

  /* ---- plot data ---- */
  const entropyBar = {
    x: entropyData?.gains ?? [],
    y: entropyData?.columns ?? [],
    type: 'bar',
    orientation: 'h',
    marker: { color: (entropyData?.gains ?? []).map(gainColor) },
    hovertemplate: '<b>%{y}</b><br>Gain: %{x:.3f} bits<extra></extra>',
  };

  const entropyLollipop = {
    x: entropyData?.gains ?? [],
    y: entropyData?.columns ?? [],
    mode: 'markers+lines',
    type: 'scatter',
    marker: { color: '#14b8a6', size: 12 },
    line: { color: '#14b8a6', width: 2 },
    hovertemplate: '<b>%{y}</b><br>Gain: %{x:.3f} bits<extra></extra>',
  };

  const sharedLayout = {
    title: 'Which categorical column splits the target best?',
    xaxis: { title: 'Information Gain (bits)' },
    yaxis: { autorange: 'reversed', type: 'category' },
    margin: { l: 180, r: 40, t: 60, b: 40 },
    height: 60 * (entropyData?.columns?.length ?? 0) || 300,
    hovermode: 'y unified',
    font: { family: 'Inter, system-ui, sans-serif' },
  };

  /* ----------  render  ---------- */
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
    >
      {/* header */}
      <div className="flex items-center space-x-2 mb-6">
        <Database className="h-5 w-5 text-blue-600" />
        <h2 className="text-xl font-semibold text-gray-900">
          Step 2: Data Review & Summary
        </h2>
      </div>

      {/* stats cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {summaryCards.map((card, idx) => {
          const Icon = card.icon;
          return (
            <div
              key={idx}
              className={`border rounded-lg p-4 ${colorClasses[card.color]}`}
            >
              <div className="flex items-center space-x-3">
                <Icon className="h-6 w-6" />
                <div>
                  <p className="text-sm font-medium opacity-80">{card.title}</p>
                  <p className="text-lg font-semibold">{card.value}</p>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* details grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* dtypes */}
        <div className="bg-gray-50 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Column Data Types</h3>
          <div className="space-y-2">
            {Object.entries(stats.dtypes).slice(0, 5).map(([col, dtype]) => (
              <div key={col} className="flex justify-between items-center">
                <span className="text-sm text-gray-600 truncate">{col}</span>
                <span className="text-xs px-2 py-1 bg-white rounded border text-gray-700">{dtype}</span>
              </div>
            ))}
            {Object.keys(stats.dtypes).length > 5 && (
              <p className="text-xs text-gray-500 italic">
                +{Object.keys(stats.dtypes).length - 5} more columns…
              </p>
            )}
          </div>
        </div>

        {/* missing */}
        <div className="bg-gray-50 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Missing Values by Column</h3>
          <div className="space-y-2">
            {Object.entries(stats.n_missing)
              .filter(([, cnt]) => cnt > 0)
              .slice(0, 5)
              .map(([col, cnt]) => (
                <div key={col} className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 truncate">{col}</span>
                  <span className="text-xs px-2 py-1 bg-amber-100 text-amber-700 rounded">{cnt}</span>
                </div>
              ))}
            {Object.entries(stats.n_missing).filter(([, cnt]) => cnt > 0).length === 0 && (
              <p className="text-sm text-green-600">✓ No missing values detected</p>
            )}
          </div>
        </div>
      </div>

      {/* numerical histograms (original) */}
      {Object.keys(numerical_data_for_plot || {}).length > 0 && (
        <div className="mt-6 bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center space-x-2 mb-4">
            <Hash className="h-5 w-5 text-indigo-600" />
            <h3 className="text-xl font-semibold text-gray-900">Numerical Feature Distributions</h3>
          </div>
          <div className="space-y-6">
            {Object.entries(numerical_data_for_plot).map(([col, data]) => (
              <div key={col} className="border border-gray-200 rounded-lg p-4">
                <Plot
                  data={[{ type: 'histogram', x: data, marker: { color: '#6366F1' } }]}
                  layout={{
                    title: `Distribution of '${col}'`,
                    xaxis: { title: col },
                    yaxis: { title: 'Count' },
                    autosize: true,
                    responsive: true,
                    font: { family: 'Inter, system-ui, sans-serif' },
                  }}
                  style={{ width: '100%', height: '300px' }}
                  config={{ responsive: true, displayModeBar: false }}
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ----------  NEW:  rich entropy section  ---------- */}
      {entropyData && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6 bg-white rounded-xl shadow-sm border border-gray-200 p-6"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-2">
              <Hash className="h-5 w-5 text-teal-600" />
              <h3 className="text-xl font-semibold text-gray-900">
                Categorical Information Gain (ID3)
              </h3>
            </div>
            <DualToggle view={view} setView={setView} />
          </div>

          <Top3Cards data={entropyData} />

          <Plot
            data={view === 'bar' ? [entropyBar] : [entropyLollipop]}
            layout={sharedLayout}
            config={{ responsive: true, displayModeBar: false }}
          />

          {/* ---- delete next line to hide heat-map ---- */}
          <EntropyHeatMap data={entropyData} />

          <p className="text-xs text-gray-500 mt-3">
            * Gain = H(Target) − H(Target|Feature). Higher is better for splitting.
          </p>
        </motion.div>
      )}
    </motion.div>
  );
};

export default DatasetSummary;