// src/components/StepNavigation.jsx
import React from 'react';

const StepNavigation = ({ currentStep, totalSteps, onNext, onBack, isNextDisabled, isStartOver }) => {
  return (
    <div className="flex justify-between items-center bg-gray-100 p-4 rounded-lg shadow-inner">
      <div className="text-sm font-medium text-gray-600">
        Step {currentStep} of {totalSteps}
      </div>
      
      <div className="space-x-3">
        {currentStep > 1 && !isStartOver && (
          <button
            onClick={onBack}
            className="px-4 py-2 bg-white border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition-colors duration-150"
          >
            ← Back
          </button>
        )}
        
        {currentStep < totalSteps && (
          <button
            onClick={onNext}
            disabled={isNextDisabled}
            className={`px-4 py-2 text-white rounded-md transition-all duration-150 ${
              isNextDisabled
                ? 'bg-blue-300 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700 shadow-md'
            }`}
          >
            {currentStep === 1 ? 'Start Processing →' : 'Next Step →'}
          </button>
        )}

        {isStartOver && (
           <button
            onClick={onNext} // Using onNext here to reset the state in App.jsx
            className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 shadow-md transition-colors duration-150"
          >
            Start Over
          </button>
        )}
      </div>
    </div>
  );
};

export default StepNavigation;// src/components/StepNavigation.jsx
