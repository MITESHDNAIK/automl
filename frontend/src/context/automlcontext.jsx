import React, { createContext, useState, useContext } from 'react';

const AutoMLContext = createContext();

export const AutoMLProvider = ({ children }) => {
  const [uploadInfo, setUploadInfo] = useState(null);
  const [trainResults, setTrainResults] = useState(null);
  const [uploadPath, setUploadPath] = useState(null);
  const [loading, setLoading] = useState(false);

  return (
    <AutoMLContext.Provider value={{
      uploadInfo, setUploadInfo,
      trainResults, setTrainResults,
      uploadPath, setUploadPath,
      loading, setLoading
    }}>
      {children}
    </AutoMLContext.Provider>
  );
};

export const useAutoML = () => useContext(AutoMLContext);