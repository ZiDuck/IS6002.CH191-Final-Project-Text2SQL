# Dataset

This directory contains Text-to-SQL datasets used for training and evaluation, available in both JSON and Parquet formats.

## Structure

```
dataset/
├── json_format/          # JSON format for inspection and analysis
│   ├── single-table_train.json
│   ├── single-table_dev.json
│   ├── bird_train.json
│   └── bird_dev.json
└── parquet_format/       # Parquet format for fine-tuning and evaluation
```

## Datasets

### Single-Table Dataset

The single-table dataset is compiled from reputable Text-to-SQL evaluation sources including WikiSQL, Spider, SQL-Create-Context, NVBench, and other public repositories. This dataset focuses on queries involving only a single table, making it ideal for models to learn direct relationships between questions and data structures.

**Statistics:**
- Training samples: 5,000
- Development samples: 200

**Schema:**
- `query`: User's natural language question
- `schema`: Database schema corresponding to the query
- `sql`: Ground truth SQL query
- `source`: Data source identifier

### BIRD Dataset

The BIRD mini_dev dataset is extracted from the original BIRD benchmark published on HuggingFace. It simulates real-world enterprise query scenarios with complex schemas, multiple table relationships, and deep reasoning requirements.

**Schema:**
- `db_id`: Database identifier
- `question`: Manually crafted question based on database content
- `evidence`: Expert-annotated domain knowledge to assist SQL generation
- `SQL`: Ground truth SQL query constructed by annotators

This subset maintains the complexity necessary to evaluate advanced query processing capabilities while being more manageable in size.

## Format Details

### JSON Format
Located in `json_format/`, these files are human-readable and suitable for data inspection, analysis, and debugging.

### Parquet Format
Located in `parquet_format/`, these files are optimized for efficient data loading during model fine-tuning and evaluation processes.

## Usage

The datasets are designed for:
- Fine-tuning Text-to-SQL models
- Evaluating model performance on single-table and multi-table queries
- Benchmarking against standard Text-to-SQL tasks
