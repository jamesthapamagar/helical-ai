import { useState, useEffect, Fragment } from 'react'

import axios from 'axios'
import Steps from './components/Steps';
import DropdownSelector from './components/Dropdown';
import DatasetSelector from './components/DatasetSelector';

function App() {
  const [step, setStep] = useState(0);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [selectedApplication, setSelectedApplication] = useState('');
  const [workflowID, setWorkflowID] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const nextStep = () => setStep((s) => Math.min(s + 1, 4 - 1));
  const prevStep = () => setStep((s) => Math.max(s - 1, 0));

  //handler for workflow
  const handleRunWorkflow = async () => {
    setLoading(true);
    setResults(null);
    try {
      //creation
      const createRes = await axios.post('/create', {
        applications: selectedApplication,
        models: selectedModel,
        datasets: selectedDataset,
      });
      const id = createRes.data.workflow_id;
      setWorkflowID(id);

      // execute 
      const execRes = await axios.post('/execute', {
        workflow_id: id
      });
      setResults(execRes.data.results);
    } catch (error) {
      setResults({ error: error.message });
    }
    setLoading(false);
  };

  return (
    <Fragment>
      <div className="flex flex-col justify-center items-center space-y-16">
        <Steps currentStep={step} />
        <div
          className="glass backdrop-blur-md shadow-lg rounded-xl flex flex-col justify-evenly items-center w-full max-w-md sm:max-w-lg md:max-w-xl lg:max-w-2xl mx-auto h-[350px] min-h-[350px] sm:h-[400px] sm:min-h-[400px]"
        >
          {step === 0 && (
            <Fragment>
              <DatasetSelector onSelect={setSelectedDataset} />
            </Fragment>
          )}
          {step === 1 && (
            <DropdownSelector
              getURL="/models"
              postURL="/select-model"
              itemKey="models"
              label="Model"
              showBatchSize={true}
              onSelect={setSelectedModel}
            />
          )}
          {step === 2 && (
            <DropdownSelector
              getURL="/applications"
              postURL="/select-application"
              itemKey="applications"
              label="Application"
              showBatchSize={false}
              onSelect={setSelectedApplication}
            />
          )}
          {step === 3 && (
            <div className="flex flex-col items-center space-y-4">
              <div className='flex flex-col justify-center items-center'>
                <p> {selectedDataset} </p>
                <p> {selectedModel} </p>
                <p> {selectedApplication} </p>
              </div>
              <button className="btn btn-primary" onClick={handleRunWorkflow} disabled={loading || !selectedDataset || !selectedModel || !selectedApplication}>
                {loading ? "Running..." : "Run Workflow"}
              </button>
              {loading ? <span className="loading loading-spinner loading-xl"></span> : ''}
              {workflowID && <div>Workflow ID: {workflowID}</div>}
              {results && (
                <div className='mt-4'>
                  <h3 className="font-bold">Results: </h3>
                  <pre>{JSON.stringify(results, null, 2)}</pre>
                </div>
              )}
            </div>
          )}

        </div>
        <div className="my-4 flex gap-2">
          <button className="btn" onClick={prevStep} disabled={step === 0}>Back</button>
          <button className="btn" onClick={nextStep} disabled={step === 4}>Next</button>
        </div>
      </div>

    </Fragment>
  )
}

export default App
