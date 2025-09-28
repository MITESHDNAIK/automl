import React, { useState } from 'react';
import { Settings, Play, HelpCircle, Info } from 'lucide-react';
import axios from 'axios';

const FineTuning = ({ uploadInfo, onTrainComplete, loading, setLoading }) => {
  const [params, setParams] = useState({
    max_depth: 5,
    n_estimators: 100,
    kernel: 'rbf',
    n_neighbors: 5,
    n_clusters: 3,
    n_components: 2,
  });
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Generate enhanced demo training results
  const generateDemoResults = () => {
    return {
      best_model: "Random Forest",
      explanation: "Based on your dataset analysis, Random Forest performed best, achieving 94.2% accuracy and 0.91 F1-macro score. Random Forest is robust ensemble method that reduces overfitting. Handles mixed data types well, provides feature importance, and works with missing values. It's particularly effective for your dataset because it provides robust predictions through ensemble learning.",
      perf_plotly: JSON.stringify({
        data: [
          {
            x: ['Decision Tree', 'Random Forest', 'Logistic Regression', 'SVM', 'KNN', 'Naive Bayes', 'XGBoost'],
            y: [0.87, 0.942, 0.89, 0.91, 0.85, 0.88, 0.935],
            type: 'bar',
            name: 'Accuracy',
            marker: { color: '#3B82F6' }
          },
          {
            x: ['Decision Tree', 'Random Forest', 'Logistic Regression', 'SVM', 'KNN', 'Naive Bayes', 'XGBoost'],
            y: [0.83, 0.91, 0.86, 0.89, 0.82, 0.85, 0.925],
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
      }),
      feature_importance_plotly: JSON.stringify({
        data: [
          {
            x: [0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05],
            y: ['credit_score', 'income', 'debt_ratio', 'age', 'experience', 'education_years', 'employment_type', 'region'],
            type: 'bar',
            orientation: 'h',
            marker: { color: 'teal' }
          }
        ],
        layout: {
          title: 'Top 8 Feature Importance - Random Forest',
          xaxis: { title: 'Importance' },
          yaxis: { title: 'Features' },
          height: 400
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
        kernel: params.kernel,
        n_neighbors: params.n_neighbors,
        n_clusters: params.n_clusters,
        n_components: params.n_components,
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
        timeout: 60000, // 60 second timeout for training multiple models
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
      [param]: param === 'kernel' ? value : (parseInt(value) || null),
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

      <div className="mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <div className="flex items-start space-x-2">
          <Info className="h-5 w-5 text-blue-600 mt-0.5 flex-shrink-0" />
          <div>
            <h3 className="font-medium text-blue-900 mb-2">Available Algorithms</h3>
            <p className="text-sm text-blue-800 mb-2">
              The system will automatically test multiple algorithms and select the best performer:
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm text-blue-700">
              <div>
                <strong>Classification:</strong>
                <ul className="list-disc list-inside ml-2">
                  <li>Decision Tree</li>
                  <li>Random Forest</li>
                  <li>Logistic Regression</li>
                  <li>Support Vector Machine</li>
                  <li>K-Nearest Neighbors</li>
                  <li>Naive Bayes</li>
                  <li>XGBoost</li>
                </ul>
              </div>
              <div>
                <strong>Regression:</strong>
                <ul className="list-disc list-inside ml-2">
                  <li>Linear Regression</li>
                  <li>Decision Tree Regressor</li>
                  <li>Random Forest Regressor</li>
                  <li>Support Vector Regression</li>
                  <li>K-Nearest Neighbors</li>
                  <li>XGBoost Regressor</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Parameter Controls */}
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Tree Max Depth
            </label>
            <input
              type="number"
              min="1"
              max="20"
              value={params.max_depth || ''}
              onChange={(e) => handleParamChange('max_depth', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="Default: Auto"
            />
            {showAdvanced && (
              <p className="text-xs text-gray-600 mt-1">
                Controls tree depth for Decision Tree and Random Forest. Lower values prevent overfitting.
              </p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Forest Estimators
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
                Number of trees in Random Forest. More trees improve accuracy but increase training time.
              </p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              SVM Kernel
            </label>
            <select
              value={params.kernel}
              onChange={(e) => handleParamChange('kernel', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="rbf">RBF (Radial Basis Function)</option>
              <option value="linear">Linear</option>
              <option value="poly">Polynomial</option>
              <option value="sigmoid">Sigmoid</option>
            </select>
            {showAdvanced && (
              <p className="text-xs text-gray-600 mt-1">
                Kernel function for SVM. RBF works well for most datasets with non-linear patterns.
              </p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              KNN Neighbors
            </label>
            <input
              type="number"
              min="1"
              max="20"
              value={params.n_neighbors || ''}
              onChange={(e) => handleParamChange('n_neighbors', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="Default: 5"
            />
            {showAdvanced && (
              <p className="text-xs text-gray-600 mt-1">
                Number of nearest neighbors for KNN algorithm. Lower values capture local patterns but may overfit.
              </p>
            )}
          </div>

          <button
            onClick={handleTrain}
            disabled={loading}
            className="w-full flex items-center justify-center space-x-2 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:bg-gray-400 text-white px-4 py-3 rounded-md font-medium transition-all duration-200 shadow-lg"
          >
            {loading ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                <span>{isDemoMode ? 'Running Demo Training...' : 'Training All Models...'}</span>
              </>
            ) : (
              <>
                <Play className="h-4 w-4" />
                <span>Train & Compare Models</span>
              </>
            )}
          </button>

          {loading && (
            <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <p className="text-sm text-blue-800 font-medium">Training Progress:</p>
              <p className="text-xs text-blue-600 mt-1">
                Testing multiple algorithms to find the best performer for your dataset...
              </p>
            </div>
          )}
        </div>

        {/* Information Panel */}
        {showAdvanced && (
          <div className="bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg p-4 border border-blue-200">
            <div className="flex items-start space-x-2">
              <Info className="h-5 w-5 text-blue-600 mt-0.5 flex-shrink-0" />
              <div>
                <h3 className="font-medium text-blue-900 mb-3">Algorithm Selection Guide</h3>
                <div className="space-y-4 text-sm">
                  <div className="bg-white rounded-lg p-3 border border-blue-100">
                    <p className="font-medium text-gray-800 mb-2">Tree-Based Models:</p>
                    <ul className="list-disc list-inside space-y-1 text-gray-700 text-xs">
                      <li><strong>Decision Tree:</strong> Interpretable but prone to overfitting</li>
                      <li><strong>Random Forest:</strong> Robust ensemble, handles mixed data well</li>
                      <li><strong>XGBoost:</strong> State-of-the-art gradient boosting</li>
                    </ul>
                  </div>
                  
                  <div className="bg-white rounded-lg p-3 border border-blue-100">
                    <p className="font-medium text-gray-800 mb-2">Linear Models:</p>
                    <ul className="list-disc list-inside space-y-1 text-gray-700 text-xs">
                      <li><strong>Linear/Logistic Regression:</strong> Fast, interpretable</li>
                      <li><strong>SVM:</strong> Powerful for high-dimensional data</li>
                    </ul>
                  </div>
                  
                  <div className="bg-white rounded-lg p-3 border border-blue-100">
                    <p className="font-medium text-gray-800 mb-2">Instance-Based:</p>
                    <ul className="list-disc list-inside space-y-1 text-gray-700 text-xs">
                      <li><strong>KNN:</strong> Simple, good for local patterns</li>
                      <li><strong>Naive Bayes:</strong> Fast probabilistic classifier</li>
                    </ul>
                  </div>
                </div>
                
                <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded">
                  <p className="text-xs text-yellow-800">
                    <strong>Tip:</strong> The system automatically tests all relevant algorithms and selects the best performer based on cross-validation metrics.
                  </p>
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