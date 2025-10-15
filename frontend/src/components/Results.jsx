// In src/components/Results.jsx
import React from 'react';
import { useAutoML } from '../context/automlcontext';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';

const Results = () => {
  const { trainResults } = useAutoML();
  const navigate = useNavigate();

  if (!trainResults || trainResults.status === 'pending') {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="bg-yellow-50 border border-yellow-200 p-6 rounded-lg text-center"
      >
        <p className="text-sm text-yellow-800">Loading results...</p>
      </motion.div>
    );
  }

  if (trainResults.status === 'failed') {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="bg-red-50 border border-red-200 p-6 rounded-lg"
      >
        <p className="text-sm text-red-800">Training failed: {trainResults.error}</p>
      </motion.div>
    );
  }

  const bestModel = trainResults.best_model;
  const results = trainResults.results;

  const handleNext = () => {
    navigate('/report');
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 max-w-4xl mx-auto"
    >
      <h2 className="text-2xl font-semibold text-gray-900 mb-6">Training Results</h2>
      <div className="space-y-4">
        <h3 className="text-lg font-medium text-gray-900">Best Model: {bestModel}</h3>
        {Object.entries(results).map(([model, metrics]) => (
          <div key={model} className="bg-gray-50 p-4 rounded-lg">
            <h4 className="text-md font-semibold">{model}</h4>
            {Object.entries(metrics).map(([metric, value]) => (
              <p key={metric} className="text-sm text-gray-700">{metric}: {value}</p>
            ))}
          </div>
        ))}
        <button
          onClick={handleNext}
          className="mt-4 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          View Report
        </button>
      </div>
    </motion.div>
  );
};

export default Results;