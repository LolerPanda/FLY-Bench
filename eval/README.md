# Flight Model Evaluation Framework

A comprehensive evaluation framework for assessing flight prediction models across multiple dimensions including accuracy, instruction following, and response quality.

## Overview

This framework provides a systematic approach to evaluate language models on flight prediction tasks through a three-stage evaluation process:

1. **Comprehensive Performance Evaluation** (`comprehensive_evaluation_clean.py`): Assesses model accuracy and success rates
2. **Instruction Following Analysis** (`instruction_following_analysis_clean.py`): Evaluates model compliance with structured output requirements  
3. **Final Ranking** (`final_ranking_clean.py`): Combines both metrics to provide overall model rankings

## Features

- **Multi-dimensional Evaluation**: Accuracy (60%) + Instruction Following (40%)
- **Robust Error Handling**: Distinguishes between API errors and instruction compliance issues
- **Structured Output Analysis**: Validates JSON format and required field completeness
- **Comprehensive Metrics**: MAE, RMSE, success rates, field completeness, and more
- **Detailed Reporting**: Generates CSV reports and console summaries
- **Proper Logging**: Uses Python logging instead of print statements
- **Internationalization**: All comments and outputs in English

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Dependencies

- Python 3.7+
- pandas>=1.3.0
- numpy>=1.21.0
- scikit-learn>=1.0.0
- matplotlib>=3.5.0
- tqdm>=4.62.0

## Usage

### Step 1: Comprehensive Performance Evaluation

```bash
python eval/comprehensive_evaluation_clean.py --results_dir your_results_dir
```

This generates `evaluation_results/comprehensive_model_ranking.csv` with accuracy metrics.

### Step 2: Instruction Following Analysis

```bash
python eval/instruction_following_analysis_clean.py
```

This generates `evaluation_results/instruction_following_analysis.csv` with instruction compliance metrics.

### Step 3: Final Ranking

```bash
python eval/final_ranking_clean.py
```

This combines the previous results and generates `evaluation_results/final_model_ranking.csv` with final rankings.

## Input Data Format

The framework expects JSONL files with the following structure:

```json
{
    "response": "model_output_text_or_json",
    "reference": {
        "Latitude (WGS84 deg)": 40.7128,
        "Longitude (WGS84 deg)": -74.0060,
        "GPS Altitude (WGS84 ft)": 1000.0
    }
}
```

## Required Flight Parameters

The framework evaluates models on 19 flight parameters:
- Latitude (WGS84 deg), Longitude (WGS84 deg), GPS Altitude (WGS84 ft)
- GPS Ground Track (deg true), Magnetic Heading (deg)
- GPS Velocity E/N/U (m/s), GPS Ground Speed (kt)
- Roll (deg), Pitch (deg), Turn Rate (deg/sec)
- Slip/Skid, Normal/Lateral Acceleration (G)
- Vertical Speed (fpm), Indicated Airspeed (kt)
- Baro Altitude (ft), Pressure Altitude (ft)

## Output Files

1. **`comprehensive_model_ranking.csv`**: Accuracy-based rankings with MAE/RMSE metrics
2. **`instruction_following_analysis.csv`**: Instruction compliance analysis
3. **`final_model_ranking.csv`**: Combined final rankings with grades (S/A/B/C/D)

## Evaluation Methodology

### Scoring Algorithm

1. **Accuracy Score** (60%): Based on MAE and RMSE performance
2. **Instruction Following Score** (40%): Weighted combination of:
   - Field Completeness (50%)
   - Value Validity (30%) 
   - Format Correctness (20%)

### Grade Classification

- **S Grade**: ≥90 points (Excellent)
- **A Grade**: 80-90 points (Good)
- **B Grade**: 70-80 points (Fair)
- **C Grade**: 60-70 points (Poor)
- **D Grade**: <60 points (Very Poor)

## Configuration

Each script accepts command-line arguments for customization:

```bash
# Comprehensive evaluation with custom paths
python eval/comprehensive_evaluation_clean.py --results_dir custom_results --output_csv custom_output.csv

# Instruction analysis with custom results directory
python eval/instruction_following_analysis_clean.py --results_dir custom_results

# Final ranking (uses files from evaluation_results/ by default)
python eval/final_ranking_clean.py
```

## Directory Structure

```
eval/
├── README_clean.md
├── comprehensive_evaluation_clean.py    # Step 1: Accuracy evaluation
├── instruction_following_analysis_clean.py  # Step 2: Instruction compliance
└── final_ranking_clean.py              # Step 3: Final rankings
```

## Logging

The framework uses Python's built-in logging module for all output. Log levels can be configured:

```python
import logging
logging.basicConfig(level=logging.DEBUG)  # For verbose output
logging.basicConfig(level=logging.WARNING)  # For minimal output
```

## Error Handling

The framework distinguishes between different types of errors:

- **API Errors**: Technical failures (not counted against instruction following)
- **Format Errors**: JSON parsing issues (counted against instruction following)
- **Field Errors**: Missing or invalid fields (counted against instruction following)

## License

MIT License

---

**Note**: This framework is designed for research and evaluation purposes. Ensure compliance with relevant regulations when using for production flight systems. 