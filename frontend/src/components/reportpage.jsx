import React, { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import { jsPDF } from 'jspdf';
import { Download, FileText, Calendar, Hash, Target, Cpu } from 'lucide-react';
import axios from 'axios';
import { useAutoML } from '../context/automlcontext';

const ReportPage = () => {
  const { state } = useLocation();
  const { trainResults: contextTrainResults, uploadPath: contextUploadPath } = useAutoML();
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchReport = async () => {
      let reportId;
      const effectiveUploadPath = state?.uploadPath || contextUploadPath;
      const effectiveTrainResults = state?.trainResults || contextTrainResults;

      if (effectiveUploadPath && effectiveTrainResults) {
        try {
          const res = await axios.post('http://localhost:8000/generate_report_data', {
            upload_path: effectiveUploadPath,
            target_column: effectiveTrainResults.dataset_stats.target,
            train_results: effectiveTrainResults,
          });
          reportId = res.data.report_id;
          localStorage.setItem('automlReportId', reportId);
        } catch (err) {
          console.error('Error generating report ID:', err);
          setLoading(false);
          return;
        }
      } else {
        reportId = localStorage.getItem('automlReportId');
        if (!reportId) {
          setLoading(false);
          return;
        }
      }

      try {
        const res = await axios.get(`http://localhost:8000/report/${reportId}`);
        setReport(res.data);
      } catch (err) {
        console.error('Error fetching report:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchReport();
  }, [state, contextTrainResults, contextUploadPath]);

  const downloadPDF = () => {
    const doc = new jsPDF();
    doc.setFontSize(18);
    doc.text('AutoML Comprehensive Report', 14, 20);
    doc.setFontSize(10);
    doc.text(`Generated: ${new Date().toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' })}`, 14, 30);
    if (report?.report_metadata) {
      doc.text(`Dataset: ${report.report_metadata.dataset_name}`, 14, 38);
      doc.text(`Target column: ${report.report_metadata.target_column}`, 14, 46);
      doc.text(`Rows: ${report.report_metadata.rows}  |  Columns: ${report.report_metadata.cols}`, 14, 54);
    }

    doc.setFontSize(12);
    doc.text('Top 5 ID3 Information Gains', 14, 70);
    doc.setFontSize(10);
    let y = 78;
    if (report?.id3_gain_analysis) {
      report.id3_gain_analysis.columns.slice(0, 5).forEach((col, idx) => {
        doc.text(`${idx + 1}. ${col} – ${report.id3_gain_analysis.gains[idx].toFixed(3)} bits`, 18, y);
        y += 6;
      });
    }

    doc.setFontSize(12);
    doc.text('Best Model', 14, y + 10);
    doc.setFontSize(10);
    if (report?.ml_analysis) {
      doc.text(`${report.ml_analysis.best_model || 'N/A'}`, 18, y + 18);
    }

    doc.save(`automl_report_${new Date().toISOString().slice(0, 10)}.pdf`);
  };

  if (loading) return <div className="p-8 text-center">Building report…</div>;
  if (!report) return <div className="p-8 text-center text-red-600">No data available.</div>;

  const MetricCard = ({ icon, label, value }) => (
    <div className="flex items-center gap-3 rounded-lg border border-gray-200 p-3">
      <div className="text-indigo-600">{icon}</div>
      <div>
        <div className="text-gray-500 text-xs">{label}</div>
        <div className="font-semibold">{value}</div>
      </div>
    </div>
  );

  const Section = ({ title, children }) => (
    <div>
      <h2 className="font-semibold text-gray-800 mb-3">{title}</h2>
      {children}
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 px-4 py-8">
      <div className="max-w-5xl mx-auto bg-white rounded-xl shadow border border-gray-200 p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <FileText className="h-6 w-6 text-indigo-600" />
            <h1 className="text-2xl font-bold text-gray-900">Comprehensive Report</h1>
          </div>
          <div className="flex gap-2">
            <button
              onClick={downloadPDF}
              className="flex items-center gap-2 px-3 py-2 text-sm rounded-md bg-indigo-600 text-white hover:bg-indigo-700"
            >
              <Download className="h-4 w-4" /> PDF
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
          <MetricCard icon={<Hash />} label="Rows" value={report.report_metadata.rows} />
          <MetricCard icon={<Target />} label="Target" value={report.report_metadata.target_column} />
          <MetricCard icon={<Cpu />} label="Best model" value={report.ml_analysis.best_model || 'N/A'} />
          <MetricCard icon={<Calendar />} label="Generated" value={new Date().toLocaleDateString('en-IN', { timeZone: 'Asia/Kolkata' })} />
        </div>

        <Section title="ID3 Information Gain (categorical features)">
          <div className="max-h-64 overflow-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="text-left p-2">Feature</th>
                  <th className="text-left p-2">Gain (bits)</th>
                </tr>
              </thead>
              <tbody>
                {report.id3_gain_analysis.columns.map((c, i) => (
                  <tr key={c} className="border-t">
                    <td className="p-2">{c}</td>
                    <td className="p-2 font-mono">{report.id3_gain_analysis.gains[i].toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>

        <Section title="Model comparison">
          <pre className="text-xs bg-gray-50 p-3 rounded overflow-auto">
            {JSON.stringify(report.ml_analysis.results, null, 2)}
          </pre>
        </Section>
      </div>
    </div>
  );
};

export default ReportPage;