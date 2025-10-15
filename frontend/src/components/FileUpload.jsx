import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, AlertCircle, CheckCircle2, Zap, Database } from 'lucide-react';
import axios from 'axios';
import { useAutoML } from '../context/automlcontext';

const FileUpload = () => {
  const { setUploadInfo, setUploadPath, loading, setLoading } = useAutoML();
  const [error, setError] = useState(null);
  const [demoMode, setDemoMode] = useState(false);
  const [uploadData, setUploadData] = useState(null);
  const [targetColumn, setTargetColumn] = useState(null);
  const [columns, setColumns] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null); // New state to store the selected file

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
      let sum = 0;
      for (let j = 0; j < alpha; j++) {
        sum += Math.log(Math.random());
      }
      let value = Math.pow(1 - Math.exp(-sum / alpha), 1 / beta) * (max - min) + min;
      data.push(value);
    }
    return data;
  };

  const onDrop = useCallback(async (acceptedFiles) => {
    if (acceptedFiles.length === 0) {
      setError('Please upload a CSV file.');
      return;
    }

    const fd = new FormData();
    fd.append('file', acceptedFiles[0]);

    setLoading(true);
    setError(null);
    setDemoMode(false);

    try {
      const res = await axios.post('http://localhost:8000/upload', fd);
      setUploadData(res.data);
      setUploadInfo(res.data);
      setUploadPath(res.data.upload_path);
      const cols = Object.keys(res.data.stats.dtypes);
      setColumns(cols);
      setTargetColumn(res.data.stats.target || cols[cols.length - 1]);
      setSelectedFile(acceptedFiles[0]); // Store the selected file
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  }, [setUploadInfo, setUploadPath, setLoading]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'] },
    maxFiles: 1
  });

  const handleTargetChange = (e) => {
    setTargetColumn(e.target.value);
  };

  const handleConfirmTarget = async () => {
    if (!targetColumn || !uploadData || !selectedFile) return;

    setLoading(true);
    const fd = new FormData();
    fd.append('file', selectedFile); // Use the stored file reference
    fd.append('target_column', targetColumn);

    try {
      const res = await axios.post('http://localhost:8000/upload', fd);
      setUploadData(res.data);
      setUploadInfo(res.data);
      setUploadPath(res.data.upload_path);
      setColumns(Object.keys(res.data.stats.dtypes));
      setError(null); // Clear any previous error
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to confirm target column. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 max-w-2xl mx-auto">
      <div className="flex items-center space-x-2 mb-6">
        <Upload className="h-6 w-6 text-blue-600" />
        <h2 className="text-xl font-semibold text-gray-900">Step 1: Upload Your Dataset</h2>
      </div>

      <div className="mb-4">
        <button
          onClick={() => {
            setDemoMode(true);
            setLoading(true);
            setTimeout(() => {
              const demoData = generateDemoData();
              setUploadData(demoData);
              setUploadInfo(demoData);
              setUploadPath(demoData.upload_path);
              setColumns(Object.keys(demoData.stats.dtypes));
              setTargetColumn(demoData.stats.target);
              setLoading(false);
            }, 2000);
          }}
          disabled={loading}
          className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-purple-500 to-blue-500 text-white rounded-md hover:from-purple-600 hover:to-blue-600 disabled:opacity-50"
        >
          <Zap className="h-4 w-4" />
          <span>Try with Demo Data</span>
        </button>
        <p className="text-xs text-gray-500 mt-2">Demo dataset contains 1,500 rows with mixed data types!</p>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4 flex items-start space-x-2">
          <AlertCircle className="h-5 w-5 text-red-500 flex-shrink-0" />
          <div>
            <p className="text-sm font-medium text-red-900">Upload Failed</p>
            <p className="text-sm text-red-800">{error}</p>
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
          ) : selectedFile ? (
            <>
              <CheckCircle2 className="h-12 w-12 text-green-500" />
              <p className="text-lg font-medium text-gray-900">File ready for upload</p>
              <p className="text-sm text-gray-600">{selectedFile.name}</p>
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

      {uploadData && (
        <div className="mt-4">
          <h3 className="text-md font-medium text-gray-900">Select Target Column</h3>
          <select
            className="w-full p-2 border border-gray-300 rounded-md mt-2"
            value={targetColumn}
            onChange={handleTargetChange}
          >
            <option value="" disabled>Select a target column</option>
            {columns.map(col => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
          <button
            onClick={handleConfirmTarget}
            className="mt-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
            disabled={!targetColumn || loading}
          >
            Confirm Target
          </button>
        </div>
      )}
    </div>
  );
};

export default FileUpload;