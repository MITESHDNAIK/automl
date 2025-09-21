import React from 'react';
import { BarChart3, Target, AlertTriangle, Database, Hash } from 'lucide-react';
import Plot from 'react-plotly.js';

const DatasetSummary = ({ uploadInfo }) => {
  const { stats, numerical_data_for_plot } = uploadInfo;

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

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <div className="flex items-center space-x-2 mb-6">
        <Database className="h-5 w-5 text-blue-600" />
        <h2 className="text-xl font-semibold text-gray-900">Dataset Summary</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {summaryCards.map((card, index) => {
          const Icon = card.icon;
          return (
            <div
              key={index}
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

      {/* Detailed Information */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Data Types */}
        <div className="bg-gray-50 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Column Data Types</h3>
          <div className="space-y-2">
            {Object.entries(stats.dtypes).slice(0, 5).map(([column, dtype]) => (
              <div key={column} className="flex justify-between items-center">
                <span className="text-sm text-gray-600 truncate">{column}</span>
                <span className="text-xs px-2 py-1 bg-white rounded border text-gray-700">
                  {dtype}
                </span>
              </div>
            ))}
            {Object.keys(stats.dtypes).length > 5 && (
              <p className="text-xs text-gray-500 italic">
                +{Object.keys(stats.dtypes).length - 5} more columns...
              </p>
            )}
          </div>
        </div>

        {/* Missing Values */}
        <div className="bg-gray-50 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Missing Values by Column</h3>
          <div className="space-y-2">
            {Object.entries(stats.n_missing)
              .filter(([, count]) => count > 0)
              .slice(0, 5)
              .map(([column, count]) => (
                <div key={column} className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 truncate">{column}</span>
                  <span className="text-xs px-2 py-1 bg-amber-100 text-amber-700 rounded">
                    {count}
                  </span>
                </div>
              ))}
            {Object.entries(stats.n_missing).filter(([, count]) => count > 0).length === 0 && (
              <p className="text-sm text-green-600">✓ No missing values detected</p>
            )}
          </div>
        </div>
      </div>

      {/* Pre-ML Plot - Added Section */}
      {Object.keys(numerical_data_for_plot || {}).length > 0 && (
        <div className="mt-6 bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center space-x-2 mb-4">
            <Hash className="h-5 w-5 text-indigo-600" />
            <h3 className="text-xl font-semibold text-gray-900">Numerical Feature Distributions</h3>
          </div>
          <div className="space-y-6">
            {Object.entries(numerical_data_for_plot).map(([column, data]) => (
              <div key={column} className="border border-gray-200 rounded-lg p-4">
                <Plot
                  data={[{
                    type: 'histogram',
                    x: data,
                    marker: { color: '#6366F1' },
                  }]}
                  layout={{
                    title: `Distribution of '${column}'`,
                    xaxis: { title: column },
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
    </div>
  );
};

export default DatasetSummary;