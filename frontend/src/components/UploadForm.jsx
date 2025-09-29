import React, {useState} from "react";
import axios from "axios";

export default function UploadForm({onUploaded}) {
  const [file, setFile] = useState(null);
  const [target, setTarget] = useState("");

  async function handleUpload(e) {
    e.preventDefault();
    if(!file) return alert("Select CSV");
    const fd = new FormData();
    fd.append("file", file);
    if (target) fd.append("target_column", target);
    const res = await axios.post('http://localhost:8000/upload', formData)
    onUploaded(res.data);
  }

  return (
    <form onSubmit={handleUpload}>
      <input type="file" accept=".csv" onChange={e=>setFile(e.target.files[0])} />
      <input placeholder="target column (optional)" value={target} onChange={e=>setTarget(e.target.value)} />
      <button type="submit">Upload CSV</button>
    </form>
  );
}
