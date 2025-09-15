import React, {useState} from "react";
import UploadForm from "./components/UploadForm";
import Results from "./components/Results";

export default function App(){
  const [uploadInfo, setUploadInfo] = useState(null);
  return (
    <div style={{padding:20}}>
      <h1>AutoML MVP</h1>
      <UploadForm onUploaded={(data)=>setUploadInfo(data)} />
      {uploadInfo && <Results uploadInfo={uploadInfo} />}
    </div>
  );
}
