// src/components/DatasetSummary.jsx
import React, { useEffect, useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';
import Plot from 'react-plotly.js';
import {
  BarChart3,
  Target,
  AlertTriangle,
  Database,
  TrendingUp,
  PieChart as PieChartIcon,
  Activity,
  Star,
  GitBranch,
  Box,
  AreaChart,
  Grid,
} from 'lucide-react';

/* ----------  Multi-Select Dropdown Component  ---------- */
const MultiSelect = ({ options, selected, onChange, label }) => {
  const [isOpen, setIsOpen] = useState(false);

  const toggleOption = (option) => {
    onChange(selected.includes(option) ? selected.filter(item => item !== option) : [...selected, option]);
  };

  return (
    <div className="relative">
      <label className="block text-sm font-medium text-gray-700 mb-2">{label}</label>
      <button 
        type="button"
        className="w-full px-3 py-2 border border-gray-300 rounded-md bg-white cursor-pointer hover:border-blue-400 transition text-left"
        onClick={() => setIsOpen(!isOpen)}
      >
        <div className="flex flex-wrap gap-1">
          {selected.length > 0 ? (
            selected.map(item => (
              <span key={item} className="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs font-medium flex items-center">
                {item}
              </span>
            ))
          ) : (
            <span className="text-gray-500 text-sm">Select...</span>
          )}
        </div>
      </button>
      {isOpen && (
        <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-md shadow-lg max-h-60 overflow-y-auto">
          {options.map(option => (
            <label key={option} className="flex items-center px-3 py-2 cursor-pointer hover:bg-blue-50">
              <input
                type="checkbox"
                checked={selected.includes(option)}
                onChange={() => toggleOption(option)}
                className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <span className="ml-3 text-sm text-gray-700">{option}</span>
            </label>
          ))}
        </div>
      )}
    </div>
  );
};

/* ----------  Visualization Recommendation Engine  ---------- */
const getRecommendedVisualization = (xCols, yCols, colTypes) => {
    const xNumerical = xCols.filter(c => colTypes[c] !== 'object');
    const yNumerical = yCols.filter(c => colTypes[c] !== 'object');
    const xCategorical = xCols.filter(c => colTypes[c] === 'object');
    const yCategorical = yCols.filter(c => colTypes[c] === 'object');

    if (xNumerical.length === 1 && yNumerical.length === 1) return { type: 'scatter', reason: 'Scatter plot is perfect for exploring relationships between two numerical variables.', icon: GitBranch };
    if (xNumerical.length >= 2 && yCols.length === 0) return { type: 'heatmap', reason: 'Heatmap is excellent for visualizing correlation patterns across multiple numerical variables.', icon: Grid };
    if (xCategorical.length === 1 && xNumerical.length === 1 && yCols.length === 0) return { type: 'box', reason: 'Box plot effectively shows numerical data distribution across categories.', icon: Box };
    if (xCategorical.length === 1 && yCols.length === 0) return { type: 'pie', reason: 'Pie chart is ideal for showing the proportions of a single categorical variable.', icon: PieChartIcon };
    if (xNumerical.length === 1 && yCols.length === 0) return { type: 'histogram', reason: 'Histogram is best for visualizing the distribution of a single numerical variable.', icon: AreaChart };
    if (xCols.length > 0) return { type: 'bar', reason: 'Bar chart is a versatile tool for comparing values across categories or over time.', icon: BarChart3 };
    return null;
};

/* ----------  Visualization Generator  ---------- */
const generateVisualization = (type, xCols, yCols, data, colTypes) => {
    if (!type) return [];
    const plots = [];
    const allSelected = [...xCols, ...yCols];

    // Helper to get data for a column, returns empty array if not found
    const getData = (col) => data[col] || [];

    try {
        switch (type) {
            case 'pie':
                allSelected.filter(c => colTypes[c] === 'object').forEach(col => {
                    const counts = getData(col).reduce((acc, val) => { acc[val] = (acc[val] || 0) + 1; return acc; }, {});
                    plots.push({ data: [{ type: 'pie', labels: Object.keys(counts), values: Object.values(counts) }], layout: { title: `Distribution of ${col}` } });
                });
                break;
            case 'histogram':
                 allSelected.filter(c => colTypes[c] !== 'object').forEach(col => {
                    plots.push({ data: [{ type: 'histogram', x: getData(col), marker: { color: '#6366F1' } }], layout: { title: `Distribution of ${col}`, xaxis: { title: col } } });
                });
                break;
            case 'scatter':
                if (xCols.length > 0 && yCols.length > 0) {
                     xCols.filter(c => colTypes[c] !== 'object').forEach(xCol => {
                        yCols.filter(c => colTypes[c] !== 'object').forEach(yCol => {
                            plots.push({ data: [{ type: 'scatter', mode: 'markers', x: getData(xCol), y: getData(yCol) }], layout: { title: `${xCol} vs ${yCol}`, xaxis: { title: xCol }, yaxis: { title: yCol } } });
                        });
                    });
                }
                break;
            case 'box':
                const cat = allSelected.find(c => colTypes[c] === 'object');
                const num = allSelected.find(c => colTypes[c] !== 'object');
                if (cat && num) {
                    plots.push({ data: [{ type: 'box', x: getData(cat), y: getData(num) }], layout: { title: `Distribution of ${num} by ${cat}` } });
                }
                break;
            case 'bar':
                 allSelected.filter(c => colTypes[c] === 'object').forEach(col => {
                    const counts = getData(col).reduce((acc, val) => { acc[val] = (acc[val] || 0) + 1; return acc; }, {});
                    plots.push({ data: [{ type: 'bar', x: Object.keys(counts), y: Object.values(counts) }], layout: { title: `Counts of ${col}` } });
                });
                break;
            case 'heatmap':
                const numCols = allSelected.filter(c => colTypes[c] !== 'object');
                if (numCols.length >= 2) {
                    const df = numCols.reduce((acc, col) => ({ ...acc, [col]: getData(col) }), {});
                    const correlations = [];
                    for (let i = 0; i < numCols.length; i++) {
                        const row = [];
                        for (let j = 0; j < numCols.length; j++) {
                            const x = df[numCols[i]];
                            const y = df[numCols[j]];
                            let sx = 0, sy = 0, sxy = 0, sx2 = 0, sy2 = 0, n = 0;
                            for (let k = 0; k < Math.min(x.length, y.length); k++) {
                                if (x[k] != null && y[k] != null) {
                                    sx += x[k]; sy += y[k]; sxy += x[k] * y[k]; sx2 += x[k]**2; sy2 += y[k]**2; n++;
                                }
                            }
                            const num = n * sxy - sx * sy;
                            const den = Math.sqrt((n * sx2 - sx**2) * (n * sy2 - sy**2));
                            row.push(den === 0 ? 0 : num / den);
                        }
                        correlations.push(row);
                    }
                    plots.push({ data: [{ type: 'heatmap', z: correlations, x: numCols, y: numCols, colorscale: 'RdBu', zmid: 0 }], layout: { title: 'Correlation Heatmap' } });
                }
                break;
        }
    } catch(error) {
        console.error("Error generating visualization:", error);
    }
    return plots.map(p => ({...p, layout: {...p.layout, autosize: true, font: { family: 'Inter, system-ui, sans-serif' }}}));
};

/* ----------  Main Component  ---------- */
const DatasetSummary = ({ uploadInfo }) => {
  if (!uploadInfo) return <div className="text-center p-8">Waiting for dataset…</div>;

  const { stats, data_for_plotting } = uploadInfo;
  const [entropyData, setEntropyData] = useState(null);
  
  const allColumns = useMemo(() => Object.keys(stats.dtypes), [stats.dtypes]);
  const [selectedXCols, setSelectedXCols] = useState([stats.target]);
  const [selectedYCols, setSelectedYCols] = useState([]);
  const [visType, setVisType] = useState('auto');
  
  const recommendation = useMemo(() => getRecommendedVisualization(selectedXCols, selectedYCols, stats.dtypes), [selectedXCols, selectedYCols, stats.dtypes]);
  const customPlots = useMemo(() => generateVisualization(visType === 'auto' ? recommendation?.type : visType, selectedXCols, selectedYCols, data_for_plotting, stats.dtypes), [visType, recommendation, selectedXCols, selectedYCols, data_for_plotting, stats.dtypes]);

  useEffect(() => {
    axios.post('http://localhost:8000/entropy_gain', { upload_path: uploadInfo.upload_path, target_column: stats.target })
      .then(res => setEntropyData(res.data)).catch(console.error);
  }, [uploadInfo.upload_path, stats.target]);

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-8">
      {/* --- Interactive Visualization Section --- */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Activity className="h-6 w-6 text-purple-600" />
          <h2 className="text-2xl font-bold text-gray-900">Interactive Data Explorer</h2>
        </div>
        
        {recommendation && (
          <div className="mb-6 p-4 bg-gradient-to-r from-yellow-50 to-orange-50 border-2 border-yellow-300 rounded-lg shadow-sm">
            <div className="flex items-start space-x-3">
              <Star className="h-7 w-7 text-yellow-500 mt-0.5 flex-shrink-0" />
              <div>
                <h4 className="font-bold text-yellow-900 text-lg mb-1">🎯 Best Suited: {recommendation.type.toUpperCase()} Chart</h4>
                <p className="text-sm text-yellow-800">{recommendation.reason}</p>
              </div>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6 pb-6 border-b border-gray-200">
          <MultiSelect options={allColumns} selected={selectedXCols} onChange={setSelectedXCols} label="X-Axis Variables" />
          <MultiSelect options={allColumns} selected={selectedYCols} onChange={setSelectedYCols} label="Y-Axis Variables (Optional)" />
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Chart Type</label>
            <select value={visType} onChange={e => setVisType(e.target.value)} className="w-full p-2 border border-gray-300 rounded-md bg-white focus:ring-2 focus:ring-purple-500">
              <option value="auto">Auto (Recommended)</option>
              <option value="bar">Bar Chart</option>
              <option value="pie">Pie Chart</option>
              <option value="histogram">Histogram</option>
              <option value="scatter">Scatter Plot</option>
              <option value="box">Box Plot</option>
              <option value="heatmap">Heatmap</option>
            </select>
          </div>
        </div>

        {customPlots.length > 0 ? (
          <div className="space-y-4">
            {customPlots.map((plot, idx) => (
              <div key={idx} className="border border-gray-200 rounded-lg p-2 bg-gray-50/50">
                <Plot data={plot.data} layout={plot.layout} style={{ width: '100%', height: '400px' }} useResizeHandler config={{ responsive: true }} />
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-12">
            <p className="text-gray-600 font-medium">Select variables to generate visualizations</p>
          </div>
        )}
      </div>

      {/* --- ID3 Information Gain Section --- */}
      {entropyData && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
           <div className="flex items-center space-x-3 mb-4">
              <TrendingUp className="h-6 w-6 text-teal-600" />
              <h3 className="text-2xl font-bold text-gray-900">Feature Influence (ID3 Analysis)</h3>
            </div>
          <div className="border border-gray-200 rounded-lg p-2">
            <Plot
              data={[{ x: entropyData.gains, y: entropyData.columns, type: 'bar', orientation: 'h', marker: {color: '#14b8a6'} }]}
              layout={{ yaxis: { autorange: 'reversed' }, margin: { l: 150 }, title: 'Which feature splits the target best?' }}
              style={{ width: '100%', height: `${Math.max(300, entropyData.columns.length * 40)}px` }}
              useResizeHandler
              config={{ responsive: true }}
            />
          </div>
        </div>
      )}
    </motion.div>
  );
};

export default DatasetSummary;
