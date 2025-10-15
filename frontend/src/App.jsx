// In src/App.jsx (partial update)
import React, { useState } from 'react';
import { Routes, Route } from 'react-router-dom';
import StepNavigation from './components/StepNavigation';
import FineTuning from './components/FineTuning';
import ResultsPage from './components/Results';
import FileUpload from './components/FileUpload';
import DatasetSummary from './components/DatasetSummary';

function App() {
  const [currentStep, setCurrentStep] = useState(1);
  const totalSteps = 4; // Upload, Dataset, Tuning, Results

  const handleNext = () => {
    if (currentStep < totalSteps) setCurrentStep(currentStep + 1);
  };

  const getPath = (step) => {
    const paths = ['', '/dataset', '/tuning', '/results'];
    return paths[step - 1];
  };

  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <StepNavigation
        currentStep={currentStep}
        totalSteps={totalSteps}
        onNext={handleNext}
        onBack={() => setCurrentStep(currentStep - 1)}
        isNextDisabled={currentStep === totalSteps}
      />
      <Routes>
        <Route path="/" element={<FileUpload />} />
        <Route path="/dataset" element={<DatasetSummary />} />
        <Route path="/tuning" element={<FineTuning />} />
        <Route path="/results" element={<ResultsPage />} />
      </Routes>
    </div>
  );
}

export default App;