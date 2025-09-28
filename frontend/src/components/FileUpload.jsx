import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, AlertCircle, CheckCircle2, Zap, Database } from 'lucide-react';
import axios from 'axios';

const FileUpload = ({ onUploadSuccess, loading, setLoading }) => {
  const [error, setError] = useState(null);
  const [demoMode, setDemoMode] = useState(false);

  // Enhanced sample demo data to showcase all algorithms
  const generateDemoData = () => {
    return {
      upload_path: "/demo/sample_dataset.csv",
      stats: {
        shape: [1500, 12],
        target: "target",
        dtypes: {
          "age": "float64",
          "income": "float64", 
          "education_years": "int64",
          "experience": "float64",
          "credit_score": "int64",
          "debt_ratio": "float64",
          "employment_type": "object",
          "marital_status": "object",
          "region": "object",
          "house_ownership": "object",
          "loan_purpose": "object",
          "target": "int64"
        },
        n_missing: {
          "age": 8,
          "income": 12,
          "education_years": 0,
          "experience": 15,
          "credit_score": 3,
          "debt_ratio": 7,
          "employment_type": 5,
          "marital_status": 2,
          "region": 0,
          "house_ownership": 4,
          "loan_purpose": 1,
          "target": 0
        }
      },
      numerical_data_for_plot: {
        "age": generateNormalDistribution(1500, 35, 12, 18, 80),
        "income": generateLogNormalDistribution(1500, 50000, 0.5),
        "credit_score": generateNormalDistribution(1500, 720, 80, 300, 850),
        "debt_ratio": generateBetaDistribution(1500, 2, 5, 0, 1)
      }
    };
  };

  // Helper functions to generate realistic demo data
  const generateNormalDistribution = (n, mean, std, min, max) => {
    const data = [];
    for (let i = 0; i < n; i++) {
      let value = (Math.random() + Math.random() + Math.random() + Math.random() + Math.random() + Math.random() - 3) / 3 * std + mean;
      value = Math.max(min, Math.min(max, value));
      data.push(Math.round(value * 100) / 100);
    }
    return data;
  };

  const generateLogNormalDistribution = (n, median, sigma) => {
    const data = [];
    for (let i = 0; i < n; i++) {
      const normal = (Math.random() + Math.random() + Math.random() + Math.random() + Math.random() + Math.random() - 3) / 3;
      const value = Math.exp(Math.log(median) + sigma * normal);
      data.push(Math.round(value));
    }
    return data;
  };

  const generateBetaDistribution = (n, alpha, beta, min, max) => {
    const data = [];
    for (let i = 0; i < n; i++) {
      // Simplified beta distribution approximation
      let sum = 0;
      for (let j = 0; j < alpha + beta; j++) {
        sum += Math.random();
      }
      const value = (sum / alpha) / ((sum / alpha) + ((alpha + beta - sum) / beta));
      const scaled = min + value * (max - min);
      data.push(Math.round(scaled * 1000) / 1000);
    }
    return data;
  };

  const onDrop = useCallback(async (acceptedFiles) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    setError(null);
    setLoading(true);

    try {
      // First try to check if backend is running
      try {
        await axios.get('http://localhost:8000/health', { timeout: 3000 });
      } catch (healthError) {
        throw new Error('BACKEND_UNAVAILABLE');
      }

      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post('http://localhost:8000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 10000, // 10 second timeout
      });
      
      onUploadSuccess(response.data);
    } catch (error) {
      console.error('Upload failed:', error);
      
      if (error.message === 'BACKEND_UNAVAILABLE') {
        setError({
          message: 'Backend Server Not Running',
          details: 'The FastAPI backend server is not running on localhost:8000. Please start the backend server first.',
          type: 'backend_down',
          solution: 'Run: python run_backend.py in your backend directory'
        });
      } else if (error.code === 'ERR_NETWORK') {
        setError({
          message: 'Network Connection Failed',
          details: 'Unable to connect to the backend server. This could be due to CORS issues or server not running.',
          type: 'network',
          solution: 'Check if backend is running on port 8000'
        });
      } else {
        setError({
          message: 'Upload Failed',
          details: error.response?.data?.detail || error.message || 'Unknown error occurred',
          type: 'upload_error'
        });
      }
    } finally {
      setLoading(false);
    }
  }, [onUploadSuccess, setLoading]);

  const handleDemoMode = () => {
    setError(null);
    setDemoMode(true);
    setLoading(true);
    
    // Simulate upload delay
    setTimeout(() => {
      onUploadSuccess(generateDemoData());
      setLoading(false);
    }, 1500);
  };

  const { getRootProps, getInputProps, isDragActive, acceptedFiles } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
    },
    multiple: false,
  });

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Upload className="h-5 w-5 text-blue-600" />
          <h2 className="text-xl font-semibold text-gray-900">Dataset Upload</h2>
        </div>
        
        <button
          onClick={handleDemoMode}
          disabled={loading}
          className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600 text-white rounded-lg font-medium transition-all duration-200 disabled:opacity-50 shadow-lg"
        >
          <Zap className="h-4 w-4" />
          <span>Try Demo</span>
        </button>
      </div>

      {/* Demo Information Panel */}
      <div className="mb-4 p-4 bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 rounded-lg">
        <div className="flex items-start space-x-2">
          <Database className="h-5 w-5 text-blue-600 mt-0.5 flex-shrink-0" />
          <div>
            <h3 className="font-medium text-blue-900 mb-2">Demo Dataset Features</h3>
            <p className="text-sm text-blue-800 mb-2">
              The demo includes a realistic financial dataset with 1,500 samples and 12 features:
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm text-blue-700">
              <div>
                <strong>Numerical Features:</strong>
                <ul className="list-disc list-inside ml-2 text-xs">
                  <li>Age, Income, Education Years</li>
                  <li>Experience, Credit Score, Debt Ratio</li>
                </ul>
              </div>
              <div>
                <strong>Categorical Features:</strong>
                <ul className="list-disc list-inside ml-2 text-xs">
                  <li>Employment Type, Marital Status</li>
                  <li>Region, House Ownership, Loan Purpose</li>
                </ul>
              </div>
            </div>
            <p className="text-xs text-blue-600 mt-2 italic">
              Perfect for testing classification algorithms with mixed data types!
            </p>
          </div>
        </div>
      </div>

      {error && (
        <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-start space-x-2">
            <AlertCircle className="h-5 w-5 text-red-500 mt-0.5 flex-shrink-0" />
            <div className="flex-1">
              <h3 className="font-medium text-red-800">{error.message}</h3>
              <p className="text-sm text-red-700 mt-1">{error.details}</p>
              
              {error.solution && (
                <div className="mt-3 p-3 bg-yellow-50 border border-yellow-200 rounded">
                  <p className="text-sm text-yellow-800 font-medium">Solution:</p>
                  <code className="text-xs text-yellow-700 bg-yellow-100 px-2 py-1 rounded mt-1 block">
                    {error.solution}
                  </code>
                </div>
              )}
              
              {error.type === 'backend_down' && (
                <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded">
                  <p className="text-sm text-blue-800 font-medium">Quick Start Guide:</p>
                  <ol className="text-sm text-blue-700 mt-1 list-decimal list-inside space-y-1">
                    <li>Navigate to your backend directory</li>
                    <li>Install dependencies: <code className="bg-blue-100 px-1 rounded">pip install -r requirements.txt</code></li>
                    <li>Start server: <code className="bg-blue-100 px-1 rounded">python run_backend.py</code></li>
                    <li>Or use: <code className="bg-blue-100 px-1 rounded">uvicorn main:app --reload</code></li>
                  </ol>
                  <p className="text-sm text-blue-600 mt-2 italic">
                    Alternatively, click "Try Demo" below to explore the interface!
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all duration-200 ${
          isDragActive
            ? 'border-blue-400 bg-blue-50'
            : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
        } ${loading ? 'pointer-events-none opacity-50' : ''}`}
      >
        <input {...getInputProps()} />
        
        <div className="flex flex-col items-center space-y-4">
          {loading ? (
            <>
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
              <p className="text-lg font-medium text-gray-900">
                {demoMode ? 'Loading demo data...' : 'Uploading and processing...'}
              </p>
              <p className="text-sm text-gray-600">This may take a few moments</p>
            </>
          ) : acceptedFiles.length > 0 ? (
            <>
              <CheckCircle2 className="h-12 w-12 text-green-500" />
              <p className="text-lg font-medium text-gray-900">File ready for upload</p>
              <p className="text-sm text-gray-600">{acceptedFiles[0].name}</p>
            </>
          ) : (
            <>
              <FileText className="h-12 w-12 text-gray-400" />
              <p className="text-lg font-medium text-gray-900">
                {isDragActive ? 'Drop your CSV file here' : 'Upload your dataset'}
              </p>
              <p className="text-sm text-gray-600">
                Drag and drop a CSV file here, or click to select
              </p>
            </>
          )}
        </div>
      </div>

      <div className="mt-4 flex items-start space-x-2">
        <AlertCircle className="h-4 w-4 text-amber-500 mt-0.5 flex-shrink-0" />
        <div className="text-sm text-gray-600">
          <p className="font-medium text-gray-700 mb-1">Requirements:</p>
          <ul className="space-y-1">
            <li>File format: CSV (.csv)</li>
            <li>Include a target column for prediction</li>
            <li>Recommended: &lt; 10MB for optimal performance</li>
            <li>Mixed data types (numerical + categorical) supported</li>
          </ul>
        </div>
      </div>

      <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg">
        <p className="text-sm text-green-800">
          <strong>New:</strong> Now supports 10+ machine learning algorithms including XGBoost, SVM, and Naive Bayes for comprehensive model comparison!
        </p>
      </div>
    </div>
  );
};

export default FileUpload;