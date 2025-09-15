import React, { useState } from 'react';
import { motion } from 'framer-motion';
import Header from './components/Header';
import FileUpload from './components/FileUpload';
import DatasetSummary from './components/DatasetSummary';
import FineTuning from './components/FineTuning';
import Results from './components/Results';
import './index.css';

function App() {
  const [uploadInfo, setUploadInfo] = useState(null);
  const [trainResults, setTrainResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUploadSuccess = (data) => {
    setUploadInfo(data);
    setTrainResults(null); // Reset previous results
  };

  const handleTrainComplete = (results) => {
    setTrainResults(results);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <Header />
      
      <main className="container mx-auto px-4 py-8 max-w-6xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="space-y-8"
        >
          {/* File Upload Section */}
          <FileUpload 
            onUploadSuccess={handleUploadSuccess}
            loading={loading}
            setLoading={setLoading}
          />

          {/* Dataset Summary Section */}
          {uploadInfo && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <DatasetSummary uploadInfo={uploadInfo} />
            </motion.div>
          )}

          {/* Fine-tuning and Training Section */}
          {uploadInfo && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.4 }}
            >
              <FineTuning 
                uploadInfo={uploadInfo}
                onTrainComplete={handleTrainComplete}
                loading={loading}
                setLoading={setLoading}
              />
            </motion.div>
          )}

          {/* Results Section */}
          {trainResults && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.6 }}
            >
              <Results trainResults={trainResults} />
            </motion.div>
          )}
        </motion.div>
      </main>
    </div>
  );
}

export default App;
