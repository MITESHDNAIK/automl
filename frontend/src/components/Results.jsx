import React from "react";
import Plot from "react-plotly.js";
import axios from "axios";

export default function Results({uploadInfo}) {
  const [trainRes, setTrainRes] = React.useState(null);
  const [params, setParams] = React.useState({max_depth: 5, n_estimators: 100});

  async function handleTrain(){
    const payload = {
      upload_path: uploadInfo.upload_path,
      target_column: uploadInfo.stats.target,
      max_depth: params.max_depth || null,
      n_estimators: params.n_estimators
    };
    const res = await axios.post("http://localhost:8000/train", payload);
    setTrainRes(res.data);
  }

  return (
    <div>
      <h3>Upload summary</h3>
      <pre>{JSON.stringify(uploadInfo.stats, null, 2)}</pre>
      <div>
        <label>max_depth</label>
        <input type="number" value={params.max_depth} onChange={e=>setParams({...params, max_depth: parseInt(e.target.value)})}/>
        <label>n_estimators</label>
        <input type="number" value={params.n_estimators} onChange={e=>setParams({...params, n_estimators: parseInt(e.target.value)})}/>
        <button onClick={handleTrain}>Train</button>
      </div>

      {trainRes && (
        <>
          <h4>Results</h4>
          <pre>{JSON.stringify(trainRes.results, null, 2)}</pre>
          <p><strong>Best:</strong> {trainRes.best_model}</p>
          <p>{trainRes.explanation}</p>

          <h4>Performance Chart</h4>
          <Plot
            data={JSON.parse(trainRes.perf_plotly).data}
            layout={JSON.parse(trainRes.perf_plotly).layout}
            style={{width: '100%', height: 400}}
          />

          <h4>Confusion Matrix</h4>
          <Plot
            data={JSON.parse(trainRes.cm_plotly).data}
            layout={JSON.parse(trainRes.cm_plotly).layout}
            style={{width: '100%', height: 400}}
          />
        </>
      )}
    </div>
  );
}
