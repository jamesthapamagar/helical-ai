import React, { useState, useEffect } from 'react';
import axios from "axios";

function DatasetSelector({ onSelect }) {
    const [datasets, setDatasets] = useState([]);
    const [selected, setSelected] = useState('');
    const [file, setFile] = useState(null);
    const [uploadStatus, setUploadStatus] = useState("");

    useEffect(() => {
        axios.get("/datasets")
            .then(res => setDatasets(res.data.datasets))
            .catch(err => {
                console.error("Failed to fetch datasets:", err);
                setUploadStatus("Failed to load datasets");
            });
    }, []);

    const handleSelect = async (e) => {
        const dataset_value = e.target.value;
        setSelected(dataset_value);
        setFile(null);
        setUploadStatus("");//clears if user uploaded their own
        if (dataset_value) {
            await axios.post("/select-dataset", { datasets: dataset_value });
            setUploadStatus(`Selected: ${dataset_value}`)
            if(onSelect) onSelect(dataset_value);
        }
    };

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
        setSelected(""); //clears selected datasets
        setUploadStatus("");
        if (onSelect) onSelect("");
    };

    const handleUpload = async () => {
        if (!file) return;
        const formData = new FormData();
        formData.append("file", file);
        try {
            const res = await axios.post("/upload-dataset", formData, {
                headers: { "Content-Type": "multipart/form-data" }
            });
            setUploadStatus(`Uploaded: ${res.data.filename}`);
            setSelected(res.data.filename);
            if (onSelect) onSelect(res.data.filename);
        } catch (err) {
            setUploadStatus("Upload failed");
        }
    };

    return (
        <div className='flex flex-col justify-center items-center space-y-4'>
            <select className="w-full max-w-xs p-2 border rounded" value={selected} onChange={handleSelect}>
                <option value="">Choose existing dataset</option>
                {datasets.map(dataset => (
                    <option key={dataset} value={dataset}>{dataset}</option>
                ))}
            </select>
            <div>
                <h2>Or</h2>
            </div>
            <input type="file" className="file-input" onChange={handleFileChange} />
            <button className="btn" onClick={handleUpload} disabled={!file}>Upload</button>
            {uploadStatus && <div>{uploadStatus}</div>}
            <div>
            </div>

        </div>
    )
}
export default DatasetSelector