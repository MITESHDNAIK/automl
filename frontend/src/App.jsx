// src/App.jsx
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import Header from './components/Header';
import FileUpload from './components/FileUpload';
import DatasetSummary from './components/DatasetSummary';
import FineTuning from './components/FineTuning';
import Results from './components/Results';
import StepNavigation from './components/StepNavigation'; // IMPORTED
import './index.css';

const TOTAL_STEPS = 4;

function App() {
  const [currentStep, setCurrentStep] = useState(1); // NEW STATE FOR NAVIGATION
  const [uploadInfo, setUploadInfo] = useState(null);
  const [trainResults, setTrainResults] = useState(null);
  const [loading, setLoading] = useState(false);

  // Navigation functions
  const nextStep = () => {
    setCurrentStep(prev => Math.min(prev + 1, TOTAL_STEPS));
  };
  
  const prevStep = () => {
    setCurrentStep(prev => Math.max(prev - 1, 1));
  };

  const handleUploadSuccess = (data) => {
    setUploadInfo(data);
    setTrainResults(null);
    nextStep(); // Advance to Step 2 (Summary)
  };

  const handleTrainComplete = (results) => {
    setTrainResults(results);
    nextStep(); // Advance to Step 4 (Results)
  };
  
  const handleStartOver = () => {
    setUploadInfo(null);
    setTrainResults(null);
    setCurrentStep(1); // Reset to Step 1 (Upload)
  };

  const renderCurrentStep = () => {
    switch (currentStep) {
      case 1:
        return (
          <FileUpload 
            onUploadSuccess={handleUploadSuccess}
            loading={loading}
            setLoading={setLoading}
          />
        );
      case 2:
        return (
          <DatasetSummary uploadInfo={uploadInfo} />
        );
      case 3:
        return (
          <FineTuning 
            uploadInfo={uploadInfo}
            onTrainComplete={handleTrainComplete}
            loading={loading}
            setLoading={setLoading}
          />
        );
      case 4:
        return (
          <Results trainResults={trainResults} />
        );
      default:
        return null;
    }
  };

  // Determine if Next button should be disabled in Step 2 (Summary)
  const isNextDisabled = currentStep === 2 && !uploadInfo;

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <Header />
      
      <main className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Step Navigation Bar */}
        <div className="mb-8">
          <StepNavigation 
            currentStep={currentStep}
            totalSteps={TOTAL_STEPS}
            onNext={currentStep === 4 ? handleStartOver : nextStep} // Use handleStartOver for the last step's button
            onBack={prevStep}
            isNextDisabled={isNextDisabled || loading}
            isStartOver={currentStep === 4}
          />
        </div>
        
        {/* Main Content Area */}
        <motion.div
          key={currentStep} // Important for motion to re-trigger animation on step change
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -50 }}
          transition={{ duration: 0.3 }}
          className="space-y-8"
        >
          {renderCurrentStep()}
        </motion.div>
      </main>
    </div>
  );
}

export default App;