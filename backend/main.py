from flask import Flask, request, jsonify 
from flask_cors import CORS
from helical import models
import pkgutil
import os

from uuid import uuid4
import importlib
import pandas as pd
import anndata as ad
import warnings 

warnings.filterwarnings("ignore")


import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.nn.functional import one_hot
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def load_dataset_file(filename):
    dataset_path = os.path.join(DATASET_FOLDER, filename)
    if filename.endswith('.csv'):
        df = pd.read_csv(dataset_path)
        adata = ad.AnnData(df)
    elif filename.endswith('.h5ad'):
        adata = ad.read_h5ad(dataset_path)
    else:
        raise ValueError("Unsupported file format: " + filename)
    return adata
        
#storing in json for now
WORKFLOWS = {}
#where fodler stored with backend
DATASET_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset'))

def list_datasets():
    # list all files in the dataset folder
    return [
        f for f in os.listdir(DATASET_FOLDER)
        if os.path.isfile(os.path.join(DATASET_FOLDER, f))
    ]

#dynamically retrieve models from package
def get_models():
    model_names = []
    for model in pkgutil.iter_modules(models.__path__):
        if model.name != "base_models" and model.name !="fine_tune":
            # print("Model:", model.name)
            model_names.append(model.name)
    return model_names

app = Flask(__name__)
CORS(app)

#CONSTANTS
MODELS = get_models() 
DATASET = list_datasets()
#only cell type anntion for now
APPLICATION = ["Cell Type Annotation", "Fine Tuning", "Gene Expression Prediction"]

#retrieving model list
@app.route('/models', methods=['GET'])
def get_models():
    return jsonify({'models': MODELS})

#receiving selected model
@app.route("/select-model", methods=['POST'])
def select_model():
    data = request.get_json()
    selected = data.get("models")
    batch_size = data.get("batch_size", 30)
    print("selected ", selected, "batch_size", batch_size)
    return jsonify({"status": "ok", "selected": selected, "batch_size": batch_size})

#retrieving dataset
@app.route('/datasets', methods=['GET'])
def get_dataset():
    return jsonify({'datasets': DATASET})

#recieving selected dataset
@app.route("/select-dataset", methods=['POST'])
def select_dataset():
    data = request.get_json()
    selected = data.get("datasets")
    print("selected", selected)
    return jsonify({"status": "ok", "selected": selected})

#uploading dataset
#TODO: validate file, so no malicious uploads
@app.route("/upload-dataset", methods=['POST'])
def upload_dataset():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    save_path = os.path.join(DATASET_FOLDER, file.filename)
    file.save(save_path)
    print(save_path)
    return jsonify({'status': 'ok', 'filename': file.filename})

#retrieving available applicaiotns
@app.route('/applications', methods=['GET'])
def get_applications():
    return jsonify({'applications': APPLICATION})

#recieving selected applicaiton
@app.route('/select-application', methods=['POST'])
def select_applciation():
    data = request.get_json()
    selected = data.get("applications")
    print("Selected", selected)
    return jsonify({"status": "ok", "selected": selected})

#creating the workflows from the selected 
@app.route("/create", methods=['POST'])
def api_create():
    data = request.get_json()
    applications = data.get("applications")
    models = data.get("models")
    datasets = data.get("datasets")
    print(data)
    print(applications)
    print(models)
    print(datasets)

    if not applications or not models or not datasets:
        return jsonify({"error": "Missing required fields"}), 400
    
    workflow_id = str(uuid4())
    WORKFLOWS[workflow_id] = {
        "applications": applications,
        "models": models,
        "datasets": datasets,
        "status": "created"
    }
    print("Created workflow ", workflow_id, WORKFLOWS[workflow_id])
    return jsonify({"workflow_id": workflow_id, "status":"created"})

#checkign what workflows has been created
@app.route("/workflows", methods=['GET'])
def list_workflow():
    return jsonify(WORKFLOWS)

#applicaitons
def run_cell_type_annotation(model_name, dataset_filename):
    #loading anndata
    try:
        adata = load_dataset_file(dataset_filename)
    except Exception as e:
        return {"error": e}
    if "gene_name" not in adata.var.columns:
        adata.var["gene_name"] = adata.var_names.astype(str)
    if "LVL1" not in adata.obs.columns:
        return {"error": "LVL1 column not found in adata.obs (required for cell type annotation)"}

    # Map model_name to class names
    model_class_map = {
        "scgpt": "scGPT",
        "geneformer": "Geneformer"
    }
    config_class_map = {
        "scgpt": "scGPTConfig",
        "geneformer": "GeneformerConfig"
    }

    try:
        model_module = importlib.import_module(f"helical.models.{model_name}")
        ModelClass = getattr(model_module, model_class_map[model_name])
        ConfigClass = getattr(model_module, config_class_map[model_name])
    except Exception as e:
        return {"error": f"Model import failed: {str(e)}"}
    
    # device = "cuda" if hasattr(importlib.util.find_spec("torch"), "cuda") else "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ConfigClass(batch_size=32, device=device)
    model = ModelClass(configurer=config)
    #can't get it to run on gpu, decreasing sample size to combat
    #limiting num of cells taking way too long on cpu
    adata = adata[:200].copy()
    
    # process data and get embeddings
    try:
        print("starting data processing")
        data = model.process_data(adata, gene_names="gene_name")
        x = model.get_embeddings(data)
        if hasattr(x, "numpy"):
            x = x.numpy()
    except Exception as e:
        return {"error": f"Model execution failed: {str(e)}"}

    # prepare labels (cell types)
    try:
        print("starting labeling")
        y = np.array(adata.obs["LVL1"].tolist())
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        num_classes = len(np.unique(y_encoded))
        y_onehot = one_hot(torch.tensor(y_encoded), num_classes).float().numpy()
    except Exception as e:
        return {"error": f"Label encoding failed: {str(e)}"}

    #train and test
    try:
        X_train, X_test, y_train, y_test = train_test_split(x, y_onehot, test_size=0.1, random_state=42)
    except Exception as e:
        return {"error": f"Train/test split failed: {str(e)}"}

    input_shape = x.shape[1]
    head_model = nn.Sequential(
        nn.Linear(input_shape, 128),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, 32),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(32, num_classes)
    ).to(device)

    # print("X_train shape:", X_train.shape)
    # print("y_train shape:", y_train.shape)
    # print("Unique train labels:", np.unique(np.argmax(y_train, axis=1)))
    # print("X_test shape:", X_test.shape)
    # print("y_test shape:", y_test.shape)
    # print("Unique test labels:", np.unique(np.argmax(y_test, axis=1)))

    # training loop (short for API)
    print("training")
    optimizer = optim.Adam(head_model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(np.argmax(y_test, axis=1), dtype=torch.long).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    head_model.train()
    for epoch in range(10):  # lower epoch range to speed up
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = head_model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # eval 
    head_model.eval()
    with torch.no_grad():
        outputs = head_model(X_test_tensor)
        y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
        y_true = y_test_tensor.cpu().numpy()
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }
    return results

#api for execution
@app.route("/execute", methods=['POST'])
def api_execute():
    data = request.get_json()
    workflow_id = data.get("workflow_id")
    workflow = WORKFLOWS.get(workflow_id)
    if not workflow:
        return jsonify({"error": "Workflow not found"}), 404

    application = workflow.get("application") or workflow.get("applications")
    model = workflow.get("model") or workflow.get("models")
    dataset = workflow.get("dataset") or workflow.get("datasets")

    if application == "Cell Type Annotation":
        results = run_cell_type_annotation(model, dataset)
        workflow["results"] = results
    else:
        return jsonify({"error": f"Application '{application}' not supported yet."}), 400

    return jsonify({"workflow_id": workflow_id, "results": results})


# checking the resutlts
@app.route("/results/<workflow_id>", methods=['GET'])
def get_results(workflow_id):
    workflow = WORKFLOWS.get(workflow_id)
    if not workflow:
        return jsonify({"error": "workflow not found"}), 404
    return jsonify({"workflow_id": workflow_id, "results": workflow.get("results", "no results")})
