import React from 'react';
import Plot from 'react-plotly.js';
import { TrendingUp, Lightbulb, Award, BarChart3, Target } from 'lucide-react';

const Results = ({ trainResults }) => {
  const { best_model, perf_plotly, confusion_plotly, feature_importance_plotly, explanation } = trainResults;

  return (
    <div className="space-y-6">
      {/* AI Recommendation Card */}
      <div className="bg-gradient-to-r from-purple-50 to-blue-50 rounded-xl border border-purple-200 p-6">
        <div className="flex items-start space-x-3">
          <div className="bg-purple-100 rounded-full p-2">
            <Lightbulb className="h-5 w-5 text-purple-600" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              🧠 AI Recommendation
            </h3>
            <p className="text-gray-700 leading-relaxed">{explanation}</p>
          </div>
        </div>
      </div>

      {/* Best Model Result */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <div className="flex items-center space-x-2 mb-4">
          <Award className="h-5 w-5 text-green-600" />
          <h2 className="text-xl font-semibold text-gray-900">Best Performing Model</h2>
        </div>
        
        <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-6">
          <div className="flex items-center space-x-3">
            <div className="bg-green-100 rounded-full p-2">
              <TrendingUp className="h-5 w-5 text-green-600" />
            </div>
            <div>
              <p className="text-lg font-semibold text-green-900">{best_model}</p>
              <p className="text-sm text-green-700">Selected as the optimal model for your dataset</p>
            </div>
          </div>
        </div>

        {/* Performance Visualization */}
        {perf_plotly && (
          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <BarChart3 className="h-5 w-5 text-blue-600" />
              <h3 className="text-lg font-semibold text-gray-900">Performance Comparison</h3>
            </div>
            
            <div className="bg-gray-50 rounded-lg p-4 mb-4">
              <p className="text-sm text-gray-700 leading-relaxed">
                This chart compares the models based on two key metrics. <strong>Accuracy</strong> is the percentage of correct predictions. 
                <strong> F1 Macro</strong> is the harmonic mean of precision and recall, providing a balanced measure of performance, 
                especially for imbalanced datasets.
              </p>
            </div>

            <div className="border border-gray-200 rounded-lg p-4">
              <Plot
                data={JSON.parse(perf_plotly).data}
                layout={{
                  ...JSON.parse(perf_plotly).layout,
                  autosize: true,
                  responsive: true,
                  font: { family: 'Inter, system-ui, sans-serif' }
                }}
                style={{ width: '100%', height: '400px' }}
                config={{ responsive: true, displayModeBar: false }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Confusion Matrix */}
      {confusion_plotly && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center space-x-2 mb-4">
            <BarChart3 className="h-5 w-5 text-purple-600" />
            <h3 className="text-lg font-semibold text-gray-900">Confusion Matrix</h3>
          </div>
          
          <div className="bg-gray-50 rounded-lg p-4 mb-4">
            <p className="text-sm text-gray-700 leading-relaxed">
              The confusion matrix shows how well the model predicts each class. Diagonal values represent correct predictions, 
              while off-diagonal values show misclassifications.
            </p>
          </div>

          <div className="border border-gray-200 rounded-lg p-4">
            <Plot
              data={JSON.parse(confusion_plotly).data}
              layout={{
                ...JSON.parse(confusion_plotly).layout,
                autosize: true,
                responsive: true,
                font: { family: 'Inter, system-ui, sans-serif' }
              }}
              style={{ width: '100%', height: '400px' }}
              config={{ responsive: true, displayModeBar: false }}
            />
          </div>
        </div>
      )}
      
      {/* Feature Importance Plot - Added Section */}
      {feature_importance_plotly && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center space-x-2 mb-4">
            <Target className="h-5 w-5 text-teal-600" />
            <h3 className="text-lg font-semibold text-gray-900">Feature Importance</h3>
          </div>
          
          <div className="bg-gray-50 rounded-lg p-4 mb-4">
            <p className="text-sm text-gray-700 leading-relaxed">
              This chart shows which features in your dataset were most important for the best model to make its predictions.
            </p>
          </div>

          <div className="border border-gray-200 rounded-lg p-4">
            <Plot
              data={JSON.parse(feature_importance_plotly).data}
              layout={{
                ...JSON.parse(feature_importance_plotly).layout,
                autosize: true,
                responsive: true,
                font: { family: 'Inter, system-ui, sans-serif' }
              }}
              style={{ width: '100%', height: '400px' }}
              config={{ responsive: true, displayModeBar: false }}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default Results;