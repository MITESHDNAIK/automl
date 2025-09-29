// src/components/Results.jsx
import React from 'react';
// FIX: Added BarChart3 to the import list
import { Award, Zap, Cpu, Settings, MessageSquare, AlertTriangle, BarChart3 } from 'lucide-react'; 
import Plot from 'react-plotly.js';

const Results = ({ trainResults }) => {
  if (!trainResults) {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 text-center text-gray-500">
        No results available. Please run the training process in Step 3.
      </div>
    );
  }

  const {
    task,
    results,
    best_model,
    perf_plotly,
    confusion_plotly,
    feature_importance_plotly,
    explanation,
    dataset_stats,
  } = trainResults;

  // Convert Python snake_case keys to readable text for display
  const metricMap = {
    accuracy: 'Accuracy',
    f1_macro: 'F1-Macro Score',
    r2: 'R² Score',
    mse: 'Mean Squared Error (MSE)',
    rmse: 'Root Mean Squared Error (RMSE)',
  };

  const formatValue = (key, value) => {
    if (key === 'accuracy') {
      return (value * 100).toFixed(1) + '%';
    }
    if (key === 'f1_macro' || key === 'r2') {
      return value.toFixed(3);
    }
    if (key === 'mse' || key === 'rmse') {
      return value.toFixed(2);
    }
    return value;
  };

  const getCardColor = (name) => {
    return name === best_model ? 'bg-indigo-600 border-indigo-700' : 'bg-gray-700 border-gray-800';
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <div className="flex items-center space-x-2 mb-4">
          <Award className="h-5 w-5 text-green-600" />
          <h2 className="text-xl font-semibold text-gray-900">Step 4: Training Results & Analysis</h2>
        </div>

        {/* Best Model Summary */}
        <div className="bg-green-50 border-2 border-green-300 rounded-xl p-4 shadow-lg mb-6">
          <div className="flex items-center space-x-3">
            <Zap className="h-6 w-6 text-green-700" />
            <h3 className="text-lg font-bold text-green-800">
              Best Performer: {best_model}
            </h3>
          </div>
          <p className="text-sm text-green-700 mt-2">{explanation}</p>
        </div>

        {/* Algorithm Comparison */}
        <div className="mb-8">
          <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
            <Cpu className="h-5 w-5 mr-2 text-blue-600" />
            Algorithm Performance Comparison ({task.toUpperCase()})
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(results).map(([name, result]) => (
              <div
                key={name}
                className={`text-white p-4 rounded-lg shadow-md transition-all ${getCardColor(name)}`}
              >
                <h4 className="font-bold text-lg mb-1">{name}</h4>
                {result.error ? (
                  <p className="text-red-300 flex items-center text-sm mt-1">
                    <AlertTriangle className="h-4 w-4 mr-1" /> Error: {result.error.substring(0, 30)}...
                  </p>
                ) : (
                  <div className="text-sm space-y-1">
                    {Object.entries(result).map(([key, value]) => {
                      // Skip internal score key
                      if (key === 'score') return null;
                      return (
                        <div key={key} className="flex justify-between">
                          <span className="font-medium">{metricMap[key] || key}:</span>
                          <span>{formatValue(key, value)}</span>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Visualizations */}
        <div className="space-y-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
            <BarChart3 className="h-5 w-5 mr-2 text-purple-600" />
            Performance Visuals
          </h3>
          
          {/* Performance Plot (R2/Accuracy vs Models) */}
          {perf_plotly && (
            <div className="border border-gray-200 rounded-lg p-4">
              <Plot
                data={JSON.parse(perf_plotly).data}
                layout={{
                  ...JSON.parse(perf_plotly).layout,
                  font: { family: 'Inter, system-ui, sans-serif' },
                  autosize: true,
                  height: 400
                }}
                style={{ width: '100%', height: '400px' }}
                config={{ responsive: true }}
              />
            </div>
          )}

          {/* Confusion Matrix (Classification only) */}
          {confusion_plotly && (
            <div className="border border-gray-200 rounded-lg p-4">
              <Plot
                data={JSON.parse(confusion_plotly).data}
                layout={{
                  ...JSON.parse(confusion_plotly).layout,
                  font: { family: 'Inter, system-ui, sans-serif' },
                  autosize: true,
                  height: 450
                }}
                style={{ width: '100%', height: '450px' }}
                config={{ responsive: true }}
              />
            </div>
          )}

          {/* Feature Importance Plot */}
          {feature_importance_plotly && (
            <div className="border border-gray-200 rounded-lg p-4">
              <Plot
                data={JSON.parse(feature_importance_plotly).data}
                layout={{
                  ...JSON.parse(feature_importance_plotly).layout,
                  font: { family: 'Inter, system-ui, sans-serif' },
                  autosize: true,
                  margin: { l: 200, r: 20, b: 50, t: 50 }, // Adjust margin for long feature names
                  height: 450,
                }}
                style={{ width: '100%', height: '450px' }}
                config={{ responsive: true }}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Results;