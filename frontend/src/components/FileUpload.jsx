import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, AlertCircle, CheckCircle2, Zap } from 'lucide-react';
import axios from 'axios';

const FileUpload = ({ onUploadSuccess, loading, setLoading }) => {
  const [error, setError] = useState(null);
  const [demoMode, setDemoMode] = useState(false);

  // Sample demo data to show the interface without backend
  const generateDemoData = () => {
    return {
      upload_path: "/demo/sample_dataset.csv",
      stats: {
        shape: [1000, 8],
        target: "target",
        dtypes: {
          "feature_1": "float64",
          "feature_2": "int64", 
          "feature_3": "object",
          "feature_4": "float64",
          "feature_5": "int64",
          "feature_6": "float64",
          "feature_7": "object",
          "target": "int64"
        },
        n_missing: {
          "feature_1": 5,
          "feature_2": 0,
          "feature_3": 12,
          "feature_4": 3,
          "feature_5": 0,
          "feature_6": 8,
          "feature_7": 2,
          "target": 0
        }
      }
    };
  };

  const onDrop = useCallback(async (acceptedFiles) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    setError(null);
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post('http://localhost:8000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 5000, // 5 second timeout
      });
      
      onUploadSuccess(response.data);
    } catch (error) {
      console.error('Upload failed:', error);
      setError({
        message: 'Backend connection failed',
        details: 'The FastAPI backend is not running on localhost:8000. You can try the demo mode to explore the interface.',
        type: 'connection'
      });
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
          className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600 text-white rounded-lg font-medium transition-all duration-200 disabled:opacity-50"
        >
          <Zap className="h-4 w-4" />
          <span>Try Demo</span>
        </button>
      </div>

      {error && (
        <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-start space-x-2">
            <AlertCircle className="h-5 w-5 text-red-500 mt-0.5 flex-shrink-0" />
            <div>
              <h3 className="font-medium text-red-800">{error.message}</h3>
              <p className="text-sm text-red-700 mt-1">{error.details}</p>
              {error.type === 'connection' && (
                <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded">
                  <p className="text-sm text-blue-800 font-medium">💡 Quick Start:</p>
                  <p className="text-sm text-blue-700">Click "Try Demo" to explore the AutoML interface with sample data!</p>
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
            <li>• File format: CSV (.csv)</li>
            <li>• Include a target column for prediction</li>
            <li>• Recommended: &lt; 10MB for optimal performance</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default FileUpload;
