import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useAutoML } from '../context/automlcontext';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const FineTuning = () => {
  const { uploadInfo } = useAutoML();
  const [params, setParams] = useState({
    max_depth: null,
    n_estimators: 100,
    kernel: 'rbf',
    n_neighbors: 5,
    n_clusters: 3,
    n_components: 2,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [trainResults, setTrainResults] = useState(null);
  const navigate = useNavigate();

  const generateDemoResults = () => ({
    status: 'completed',
    best_model: 'Random Forest',
    results: {
      'Linear Regression': { 'R²': 0.6, RMSE: 1.8 },
      'Decision Tree': { 'R²': 0.7, RMSE: 1.6 },
      'Random Forest': { 'R²': 0.8, RMSE: 1.4 },
      SVR: { 'R²': 0.5, RMSE: 2.0 },
      KNN: { 'R²': 0.65, RMSE: 1.7 },
    },
  });

  const handleParamChange = (e) => {
    const { name, value } = e.target;
    setParams((prev) => ({
      ...prev,
      [name]: value === '' ? null : value,
    }));
  };

  const handleTrain = async () => {
  if (!uploadInfo) {
    setError('No dataset uploaded. Please upload a dataset in Step 1.');
    return;
  }
  setLoading(true);
  setError(null);

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

    if (uploadInfo.upload_path.includes('/demo/')) {
      setTimeout(() => {
        const demoResults = generateDemoResults();
        setTrainResults(demoResults);
        setLoading(false);
        navigate('/results'); // Navigate after demo
      }, 3000);
      return;
    }

    const trainResponse = await axios.post('http://localhost:8000/train', payload, { timeout: 60000 });
    const taskId = trainResponse.data.task_id;

    let attempts = 0;
    const maxAttempts = 12; // 60 seconds with 5-second intervals
    const pollInterval = setInterval(async () => {
      attempts++;
      try {
        const resultResponse = await axios.get(`http://localhost:8000/train_result/${taskId}`);
        if (resultResponse.data.status !== 'pending') {
          clearInterval(pollInterval);
          setTrainResults(resultResponse.data);
          setLoading(false);
          console.log('Navigating to /results with:', resultResponse.data); // Debug
          navigate('/results'); // Automatic navigation
        }
      } catch (error) {
        if (attempts >= maxAttempts) {
          clearInterval(pollInterval);
          setError('Training timed out or failed. Check backend logs.');
          setLoading(false);
        }
      }
    }, 5000);

  } catch (error) {
    setError(error.response?.data?.detail || 'Training failed. Check backend logs.');
    setLoading(false);
    if (error.code === 'ERR_NETWORK' || error.code === 'ECONNREFUSED') {
      setTimeout(() => {
        const demoResults = generateDemoResults();
        setTrainResults(demoResults);
        setLoading(false);
        navigate('/results'); // Navigate after demo fallback
      }, 3000);
    }
  }
};
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 max-w-4xl mx-auto"
    >
      <h2 className="text-2xl font-semibold text-gray-900 mb-6">Fine-Tuning</h2>
      {error && <p className="text-red-600 mb-4">{error}</p>}
      {loading && <p className="text-blue-600 mb-4">Training in progress...</p>}
      {!loading && !error && trainResults && (
        <p className="text-green-600 mb-4">Training completed! Results saved.</p>
      )}
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {Object.entries(params).map(([key, value]) => (
            <div key={key}>
              <label className="block text-sm font-medium text-gray-700">{key.replace('_', ' ')}</label>
              <input
                type={key.includes('depth') || key.includes('estimators') || key.includes('neighbors') || key.includes('clusters') || key.includes('components') ? 'number' : 'text'}
                name={key}
                value={value}
                onChange={handleParamChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                disabled={loading}
              />
            </div>
          ))}
        </div>
        <button
          onClick={handleTrain}
          disabled={loading}
          className="w-full md:w-auto px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-blue-400 disabled:cursor-not-allowed"
        >
          Train & Compare Models
        </button>
      </div>
    </motion.div>
  );
};

export default FineTuning;