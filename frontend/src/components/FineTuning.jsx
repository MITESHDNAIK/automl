import React, { useState } from 'react';
import { Settings, Play, HelpCircle, Info } from 'lucide-react';
import axios from 'axios';

const FineTuning = ({ uploadInfo, onTrainComplete, loading, setLoading }) => {
  const [params, setParams] = useState({
    max_depth: 5,
    n_estimators: 100,
  });
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Generate demo training results
  const generateDemoResults = () => {
    return {
      best_model: "Random Forest",
      explanation: "Based on your dataset analysis, Random Forest performed best with an accuracy of 94.2% and F1-macro score of 0.91. The model shows excellent performance across all classes with minimal overfitting. Random Forest is particularly effective for your dataset because it handles mixed data types well and provides robust predictions through ensemble learning.",
      perf_plotly: JSON.stringify({
        data: [
          {
            x: ['Decision Tree', 'Random Forest'],
            y: [0.87, 0.942],
            type: 'bar',
            name: 'Accuracy',
            marker: { color: '#3B82F6' }
          },
          {
            x: ['Decision Tree', 'Random Forest'],
            y: [0.83, 0.91],
            type: 'bar',
            name: 'F1 Macro',
            marker: { color: '#8B5CF6' }
          }
        ],
        layout: {
          title: 'Model Performance Comparison',
          xaxis: { title: 'Model' },
          yaxis: { title: 'Score', range: [0, 1] },
          barmode: 'group'
        }
      }),
      confusion_plotly: JSON.stringify({
        data: [
          {
            z: [[85, 5, 3], [4, 92, 1], [2, 3, 88]],
            type: 'heatmap',
            colorscale: 'Blues',
            showscale: true
          }
        ],
        layout: {
          title: 'Confusion Matrix - Random Forest',
          xaxis: { title: 'Predicted Class' },
          yaxis: { title: 'True Class' }
        }
      })
    };
  };

  const handleTrain = async () => {
    setLoading(true);
    
    try {
      const payload = {
        upload_path: uploadInfo.upload_path,
        target_column: uploadInfo.stats.target,
        max_depth: params.max_depth || null,
        n_estimators: params.n_estimators,
      };

      // Check if this is demo mode (demo upload path)
      if (uploadInfo.upload_path.includes('/demo/')) {
        // Use demo mode with simulated delay
        setTimeout(() => {
          onTrainComplete(generateDemoResults());
          setLoading(false);
        }, 3000);
        return;
      }

      // Try real backend
      const response = await axios.post('http://localhost:8000/train', payload, {
        timeout: 30000, // 30 second timeout for training
      });
      onTrainComplete(response.data);
    } catch (error) {
      console.error('Training failed:', error);
      
      // If backend fails, offer demo results
      if (error.code === 'ERR_NETWORK' || error.code === 'ECONNREFUSED') {
        alert('Backend not available. Showing demo results to showcase the interface.');
        setTimeout(() => {
          onTrainComplete(generateDemoResults());
          setLoading(false);
        }, 2000);
        return;
      }
      
      alert('Training failed. Please check the backend connection and try again.');
    } finally {
      if (!uploadInfo.upload_path.includes('/demo/')) {
        setLoading(false);
      }
    }
  };

  const handleParamChange = (param, value) => {
    setParams(prev => ({
      ...prev,
      [param]: parseInt(value) || null,
    }));
  };

  const isDemoMode = uploadInfo.upload_path.includes('/demo/');

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <Settings className="h-5 w-5 text-blue-600" />
          <h2 className="text-xl font-semibold text-gray-900">Model Configuration</h2>
          {isDemoMode && (
            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
              Demo Mode
            </span>
          )}
        </div>
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center space-x-1 text-sm text-blue-600 hover:text-blue-700"
        >
          <HelpCircle className="h-4 w-4" />
          <span>{showAdvanced ? 'Hide' : 'Show'} Help</span>
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Parameter Controls */}
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Decision Tree Max Depth
            </label>
            <input
              type="number"
              min="1"
              max="20"
              value={params.max_depth || ''}
              onChange={(e) => handleParamChange('max_depth', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="Default: 5"
            />
            {showAdvanced && (
              <p className="text-xs text-gray-600 mt-1">
                Controls how deep the decision tree can grow. Lower values prevent overfitting.
              </p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Random Forest Estimators
            </label>
            <input
              type="number"
              min="10"
              max="500"
              value={params.n_estimators || ''}
              onChange={(e) => handleParamChange('n_estimators', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="Default: 100"
            />
            {showAdvanced && (
              <p className="text-xs text-gray-600 mt-1">
                Number of trees in the random forest. More trees usually improve accuracy but increase training time.
              </p>
            )}
          </div>

          <button
            onClick={handleTrain}
            disabled={loading}
            className="w-full flex items-center justify-center space-x-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white px-4 py-3 rounded-md font-medium transition-colors duration-200"
          >
            {loading ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                <span>{isDemoMode ? 'Running Demo Training...' : 'Training Models...'}</span>
              </>
            ) : (
              <>
                <Play className="h-4 w-4" />
                <span>Train Models</span>
              </>
            )}
          </button>
        </div>

        {/* Information Panel */}
        {showAdvanced && (
          <div className="bg-blue-50 rounded-lg p-4">
            <div className="flex items-start space-x-2">
              <Info className="h-5 w-5 text-blue-600 mt-0.5 flex-shrink-0" />
              <div>
                <h3 className="font-medium text-blue-900 mb-2">Hyperparameter Tuning Guide</h3>
                <div className="space-y-3 text-sm text-blue-800">
                  <div>
                    <p className="font-medium">Max Depth (Decision Tree):</p>
                    <ul className="list-disc list-inside space-y-1 text-blue-700">
                      <li>Lower values (3-5): Prevent overfitting, good for small datasets</li>
                      <li>Higher values (10+): Capture complex patterns, risk overfitting</li>
                    </ul>
                  </div>
                  <div>
                    <p className="font-medium">N Estimators (Random Forest):</p>
                    <ul className="list-disc list-inside space-y-1 text-blue-700">
                      <li>Start with 100 for most datasets</li>
                      <li>Increase to 200-500 for better accuracy (slower training)</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default FineTuning;
