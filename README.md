# FLY-Bench: Comprehensive Fixed-Wing Aircraft Flight Data Analysis and Model Evaluation Framework

FLY-Bench is an open-source toolkit for analyzing, visualizing, and benchmarking Fixed-Wing Aircraft flight data, with a comprehensive evaluation framework for assessing flight prediction models across multiple dimensions.

## Overview

This project provides two main components:

1. **Flight Data Analysis Toolkit**: Load, process, visualize, and analyze multi-sensor Fixed-Wing Aircraft flight data
2. **Model Evaluation Framework**: Comprehensive evaluation system for assessing language models on flight prediction tasks

## Features

### Flight Data Analysis
- Load and process multi-sensor Fixed-Wing Aircraft flight data (GPS, IMU, barometer, battery, etc.)
- Visualize flight trajectories and sensor signals
- Compute and benchmark key flight performance metrics
- Modular and extensible codebase for custom analysis

### Model Evaluation Framework
- **Multi-dimensional Evaluation**: Accuracy (60%) + Instruction Following (40%)
- **Robust Error Handling**: Distinguishes between API errors and instruction compliance issues
- **Structured Output Analysis**: Validates JSON format and required field completeness
- **Comprehensive Metrics**: MAE, RMSE, success rates, field completeness, and more
- **Detailed Reporting**: Generates CSV reports and console summaries

## Dataset

The toolkit is built around the `selected_five_legs_data_v5.jsonl` dataset, which contains rich multi-sensor flight records from real Fixed-Wing Aircraft missions. Each record includes:

### Flight Parameters (19 core fields)
- **Position**: Latitude (WGS84 deg), Longitude (WGS84 deg), GPS Altitude (WGS84 ft)
- **Navigation**: GPS Ground Track (deg true), Magnetic Heading (deg)
- **Velocity**: GPS Velocity E/N/U (m/s), GPS Ground Speed (kt)
- **Attitude**: Roll (deg), Pitch (deg), Turn Rate (deg/sec)
- **Performance**: Slip/Skid, Normal/Lateral Acceleration (G)
- **Flight Data**: Vertical Speed (fpm), Indicated Airspeed (kt)
- **Altitude**: Baro Altitude (ft), Pressure Altitude (ft)

### Additional Sensor Data
- GPS status, satellites, HDOP/VDOP
- Battery status, charge percentage, supply voltage
- Attitude status, magnetic variation
- Internal temperature, barometric setting
- Distance to airport calculations

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies
- Python 3.7+
- pandas>=1.3.0
- numpy>=1.21.0
- matplotlib>=3.5.0
- scikit-learn>=1.0.0
- tqdm>=4.62.0
- colorama>=0.4.4

## Usage

### Flight Data Analysis

```bash
# Basic analysis and visualization
python main.py --data selected_five_legs_data_v5.jsonl
```

### Model Evaluation Framework

The evaluation framework provides a three-stage evaluation process:

#### Step 1: Comprehensive Performance Evaluation
```bash
python eval/comprehensive_evaluation.py --results_dir your_results_dir
```
Generates `evaluation_results/comprehensive_model_ranking.csv` with accuracy metrics.

#### Step 2: Instruction Following Analysis
```bash
python eval/instruction_following_analysis.py
```
Generates `evaluation_results/instruction_following_analysis.csv` with instruction compliance metrics.

#### Step 3: Final Ranking
```bash
python eval/final_ranking.py
```
Combines previous results and generates `evaluation_results/final_model_ranking.csv` with final rankings.

## Project Structure

```
FLY-Bench/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── main.py                            # Entry point for data analysis
├── data_loader.py                     # Data loading utilities
├── visualization.py                   # Flight trajectory visualization
├── selected_five_legs_data_v5.jsonl   # Example dataset (709 records)
└── eval/                              # Model evaluation framework
    ├── README.md                      # Evaluation framework documentation
    ├── comprehensive_evaluation.py    # Step 1: Accuracy evaluation
    ├── instruction_following_analysis.py  # Step 2: Instruction compliance
    └── final_ranking.py              # Step 3: Final rankings
```

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

## Input Data Format

### Flight Data (JSONL)
Each line contains a JSON object with flight parameters:
```json
{
    "Latitude (WGS84 deg)": "40.3670006",
    "Longitude (WGS84 deg)": "115.9680328",
    "GPS Altitude (WGS84 ft)": "2478",
    "GPS Ground Speed (kt)": "48.5",
    "Roll (deg)": "19.64",
    "Pitch (deg)": "-1.59",
    "distance_to_airport_km": 3.217
}
```

### Model Evaluation Data (JSONL)
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

## Output Files

### Analysis Results
- Flight trajectory plots (latitude vs longitude)
- Flight performance metrics (distance, speed, altitude statistics)

### Evaluation Results
1. **`comprehensive_model_ranking.csv`**: Accuracy-based rankings with MAE/RMSE metrics
2. **`instruction_following_analysis.csv`**: Instruction compliance analysis
3. **`final_model_ranking.csv`**: Combined final rankings with grades (S/A/B/C/D)

## Configuration

Each evaluation script accepts command-line arguments for customization:

```bash
# Comprehensive evaluation with custom paths
python eval/comprehensive_evaluation.py --results_dir custom_results --output_csv custom_output.csv

# Instruction analysis with custom results directory
python eval/instruction_following_analysis.py --results_dir custom_results

# Final ranking (uses files from evaluation_results/ by default)
python eval/final_ranking.py
```

## Error Handling

The evaluation framework distinguishes between different types of errors:

- **API Errors**: Technical failures (not counted against instruction following)
- **Format Errors**: JSON parsing issues (counted against instruction following)
- **Field Errors**: Missing or invalid fields (counted against instruction following)

## Dataset Statistics

The `selected_five_legs_data_v5.jsonl` dataset contains:
- **709 flight records** from real Fixed-Wing Aircraft missions
- **19 core flight parameters** for evaluation
- **Additional sensor data** for comprehensive analysis
- **Geographic coverage**: Various flight patterns and conditions
- **Time range**: Multiple flight sessions with different environmental conditions

## License

This project is licensed under the MIT License.

---

**Note**: This framework is designed for research and evaluation purposes. Ensure compliance with relevant regulations when using for production flight systems. 
