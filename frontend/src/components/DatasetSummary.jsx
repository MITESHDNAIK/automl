import React, { useEffect } from 'react';
import { motion } from 'framer-motion';
import { BarChart2, PieChart, Info } from 'lucide-react';
import { useAutoML } from '../context/automlcontext';

const Top3Cards = ({ uploadInfo }) => {
  // Fallback to empty object if dtypes is undefined
  const dtypes = uploadInfo?.stats?.dtypes || {};
  const top3Features = Object.entries(dtypes)
    .sort(([, a], [, b]) => (a === 'object' ? 1 : -1)) // Prioritize non-object types
    .slice(0, 3)
    .map(([feature]) => feature);

  if (!top3Features.length) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="bg-yellow-50 border border-yellow-200 p-4 rounded-lg"
      >
        <p className="text-sm text-yellow-800">No features available</p>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="grid grid-cols-1 md:grid-cols-3 gap-4"
    >
      {top3Features.map((feature, index) => (
        <motion.div
          key={index}
          initial={{ scale: 0.9 }}
          animate={{ scale: 1 }}
          className="bg-white border border-gray-200 p-4 rounded-lg shadow-sm"
        >
          <h3 className="text-lg font-semibold text-gray-900">{feature}</h3>
          <p className="text-sm text-gray-600">Type: {dtypes[feature]}</p>
        </motion.div>
      ))}
    </motion.div>
  );
};

const EntropyHeatMap = ({ uploadInfo }) => {
  // Fallback to empty object if numerical_data_for_plot is undefined
  const data = uploadInfo?.numerical_data_for_plot || {};
  const matrix = Object.values(data).length
    ? Object.values(data).map(col => col.map(val => val || 0)) // Handle null with 0
    : [];

  if (!matrix.length) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="bg-yellow-50 border border-yellow-200 p-4 rounded-lg"
      >
        <p className="text-sm text-yellow-800">No data for heatmap</p>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="grid grid-cols-10 gap-1"
    >
      {matrix.map((row, i) => (
        <div key={i} className="flex">
          {row.map((val, j) => (
            <div
              key={j}
              className="w-6 h-6 bg-blue-200"
              style={{ opacity: val ? val / 100 : 0.1 }} // Simple visualization
            />
          ))}
        </div>
      ))}
    </motion.div>
  );
};

const DatasetSummary = () => {
  const { uploadInfo } = useAutoML();

  useEffect(() => {
    console.log('uploadInfo:', uploadInfo);
  }, [uploadInfo]);

  if (!uploadInfo) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="bg-yellow-50 border border-yellow-200 p-6 rounded-lg text-center"
      >
        <p className="text-sm text-yellow-800">No dataset uploaded yet. Please upload a dataset.</p>
      </motion.div>
    );
  }

  const { shape, target, n_missing } = uploadInfo.stats || {};
  const missingCount = Object.values(n_missing || {}).reduce((a, b) => a + b, 0);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 max-w-4xl mx-auto"
    >
      <div className="flex items-center space-x-2 mb-6">
        <BarChart2 className="h-6 w-6 text-blue-600" />
        <h2 className="text-2xl font-semibold text-gray-900">Dataset Summary</h2>
      </div>

      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <motion.div
            initial={{ y: 20 }}
            animate={{ y: 0 }}
            className="bg-gradient-to-br from-blue-50 to-purple-50 p-4 rounded-lg border border-blue-200"
          >
            <h3 className="text-lg font-medium text-blue-900 mb-2">Overview</h3>
            <p className="text-sm text-gray-700">
              Rows: {shape?.[0] || 0}, Columns: {shape?.[1] || 0}
            </p>
            <p className="text-sm text-gray-700">Target: {target || 'N/A'}</p>
            <p className="text-sm text-gray-700">Missing Values: {missingCount}</p>
          </motion.div>

          <Top3Cards uploadInfo={uploadInfo} />
        </div>

        <div className="border-t border-gray-200 pt-4">
          <h3 className="text-lg font-medium text-gray-900 mb-2 flex items-center space-x-2">
            <PieChart className="h-5 w-5 text-purple-600" />
            <span>Entropy Heatmap</span>
          </h3>
          <EntropyHeatMap uploadInfo={uploadInfo} />
        </div>

        <div className="bg-gradient-to-br from-yellow-50 to-blue-50 p-4 rounded-lg border border-yellow-200">
          <div className="flex items-start space-x-2">
            <Info className="h-5 w-5 text-blue-600 mt-0.5 flex-shrink-0" />
            <div>
              <h4 className="font-medium text-blue-900 mb-2">Data Insights</h4>
              <p className="text-sm text-gray-700">
                This summary provides an overview of your dataset. Upload a new file or adjust
                the target column to explore different analyses.
              </p>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default DatasetSummary;