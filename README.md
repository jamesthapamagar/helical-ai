# Helical-AI Workflow Web Application

A modern web application for single-cell genomics analysis, featuring automated cell type annotation using state-of-the-art machine learning models.

## Architecture

- **Frontend**: React
- **Backend**: Flask
- **Containerisation**: Docker Compose for easy deployment and development

## Dataset Setup

**Important**: You must download and place your dataset files before running the application.

1. Create a `dataset` folder in the project root directory:
   ```bash
   mkdir dataset
   ```
2. Download your single-cell dataset file

3. Place the dataset file in the `dataset` folder:
   ```
   helical-ai/
   ├── dataset/
   │   └── your_dataset_file.h5ad
   ├── backend/
   ├── frontend/
   └── docker-compose.yaml
   ```

### 1. Clone the Repository

```bash
git clone https://github.com/jamesthapamagar/helical-ai.git
cd helical-ai
```

### 3. Build and Run the Application

docker-compose up --build

### Step-by-Step Workflow

1. **Select Dataset**: Choose from available datasets or upload your own
2. **Choose Model**: Select a pre-trained model for analysis
3. **Select Application**: Pick the type of analysis (e.g., Cell Type Annotation)
4. **Run Workflow**: Execute the analysis and view results
