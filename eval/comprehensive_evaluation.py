#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import glob
import argparse
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from collections import defaultdict
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define numeric fields for evaluation
NUMERIC_FIELDS = [
    "Latitude (WGS84 deg)", "Longitude (WGS84 deg)", "GPS Altitude (WGS84 ft)",
    "GPS Ground Track (deg true)", "Magnetic Heading (deg)", 
    "GPS Velocity E (m/s)", "GPS Velocity N (m/s)", "GPS Velocity U (m/s)",
    "GPS Ground Speed (kt)", "Roll (deg)", "Pitch (deg)", 
    "Turn Rate (deg/sec)", "Normal Acceleration (G)",
    "Lateral Acceleration (G)", "Vertical Speed (fpm)",
    "Indicated Airspeed (kt)", "Baro Altitude (ft)", "Pressure Altitude (ft)"
]

# Define special fields that require circular handling (angles)
ANGLE_FIELDS = [
    "GPS Ground Track (deg true)", "Magnetic Heading (deg)", "Roll (deg)", "Pitch (deg)"
]

def load_jsonl(file_path):
    """Load JSONL file and return data list"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Cannot parse JSON line: {line[:50]}...")
        return data
    except Exception as e:
        logger.error(f"Failed to load file {file_path}: {str(e)}")
        return []

def extract_actual_values_from_reference(item):
    """Extract actual values from reference field"""
    try:
        # Try to get reference data directly
        if 'reference' in item:
            return item['reference']
        
        # Fallback: extract from question text (for backward compatibility)
        question_text = item.get('question', '')
        if not question_text:
            return None
            
        # Look for "next second data:" section
        if "next second data:" in question_text.lower():
            next_start = question_text.lower().find("next second data:")
            next_section = question_text[next_start:]
            json_start = next_section.find('{')
            json_end = next_section.find('}')
            if json_start >= 0 and json_end > json_start:
                json_str = next_section[json_start:json_end+1]
                return json.loads(json_str)
        
        # Try to find the second JSON object
        json_objects = []
        start_pos = 0
        while True:
            json_start = question_text.find('{', start_pos)
            if json_start == -1:
                break
            
            # Find corresponding closing bracket
            brace_count = 0
            json_end = json_start
            for i in range(json_start, len(question_text)):
                if question_text[i] == '{':
                    brace_count += 1
                elif question_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i
                        break
            
            if brace_count == 0:
                try:
                    json_str = question_text[json_start:json_end+1]
                    json_obj = json.loads(json_str)
                    json_objects.append(json_obj)
                except:
                    pass
            
            start_pos = json_end + 1
        
        # If multiple JSON objects found, return the second one (next second data)
        if len(json_objects) >= 2:
            return json_objects[1]
        elif len(json_objects) == 1:
            return json_objects[0]
            
    except Exception as e:
        pass
    
    return None

def extract_predicted_values_from_response(response_text):
    """Extract predicted values from response text"""
    try:
        if isinstance(response_text, dict):
            return response_text
        
        if isinstance(response_text, str):
            # Check for API errors
            if any(error in response_text.lower() for error in ['api', 'timeout', 'error', '403', '500']):
                return None
            
            # Try to parse the entire response directly
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract JSON object from text
                json_start = response_text.find('{')
                json_end = response_text.rfind('}')
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end+1]
                    try:
                        return json.loads(json_str)
                    except:
                        # Try to extract JSON from code block
                        code_block_start = response_text.find('```')
                        if code_block_start >= 0:
                            code_block_end = response_text.find('```', code_block_start + 3)
                            if code_block_end > code_block_start:
                                code_block = response_text[code_block_start+3:code_block_end].strip()
                                # Remove first line if it starts with json/JSON
                                if code_block.startswith(('json', 'JSON')):
                                    code_block = '\n'.join(code_block.split('\n')[1:])
                                
                                json_start = code_block.find('{')
                                json_end = code_block.rfind('}')
                                if json_start >= 0 and json_end > json_start:
                                    json_str = code_block[json_start:json_end+1]
                                    try:
                                        return json.loads(json_str)
                                    except:
                                        pass
    except Exception as e:
        pass
    
    return None

def check_response_quality(item):
    """Check response quality, return boolean and details"""
    # Check if response exists
    if 'response' not in item:
        return False, "API Error", "Response does not exist"
    
    response = item['response']
    
    # Check for API errors
    if isinstance(response, str):
        if any(error in response.lower() for error in ['api', 'timeout', 'error', '403', '500']):
            return False, "API Error", response[:100]
    
    # Try to extract predicted values
    predicted_values = extract_predicted_values_from_response(response)
    if predicted_values is None:
        return False, "Format Error", "Cannot parse JSON response"
    
    # Check if required fields exist
    missing_fields = [field for field in NUMERIC_FIELDS if field not in predicted_values]
    if missing_fields:
        return False, "Missing Fields", f"Missing {len(missing_fields)} fields: {', '.join(missing_fields[:3])}..."
    
    # Check if field values are valid
    invalid_fields = []
    for field in NUMERIC_FIELDS:
        value = predicted_values.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            invalid_fields.append(field)
    
    if invalid_fields:
        return False, "Invalid Values", f"Invalid fields {len(invalid_fields)}: {', '.join(invalid_fields[:3])}..."
    
    return True, "Success", "Response is complete and valid"

def angle_error(pred, true):
    """Calculate angle error, considering circular nature (0-360 degrees)"""
    # Ensure angle is within 0-360 range
    pred = pred % 360
    true = true % 360
    
    # Calculate the difference
    diff = abs(pred - true)
    
    # Consider circular nature (e.g., 359° and 1° are only 2° apart)
    if diff > 180:
        diff = 360 - diff
    
    return diff

def evaluate_numeric_predictions(actual, predicted, field_name):
    """Evaluate numeric predictions for a specific field"""
    try:
        # Convert to numeric values
        if isinstance(actual, str):
            actual_val = float(actual)
        else:
            actual_val = float(actual)
        
        if isinstance(predicted, str):
            predicted_val = float(predicted)
        else:
            predicted_val = float(predicted)
        
        # Check for NaN or infinite values
        if np.isnan(actual_val) or np.isinf(actual_val) or np.isnan(predicted_val) or np.isinf(predicted_val):
            return None, None, None
        
        # Use angle error for angle fields
        if field_name in ANGLE_FIELDS:
            error = angle_error(predicted_val, actual_val)
        else:
            error = abs(predicted_val - actual_val)
        
        return error, actual_val, predicted_val
        
    except (ValueError, TypeError) as e:
        logger.debug(f"Evaluation error ({field_name}): {str(e)}")
        return None, None, None

def evaluate_single_model(file_path):
    """Evaluate a single model's performance"""
    model_name = os.path.basename(file_path).replace('.jsonl', '')
    data = load_jsonl(file_path)
    
    if not data:
        return None
    
    # Initialize metrics
    field_errors = defaultdict(list)
    field_actuals = defaultdict(list)
    field_predictions = defaultdict(list)
    
    success_count = 0
    total_count = len(data)
    
    for item in data:
        # Check response quality
        is_valid, error_type, error_details = check_response_quality(item)
        
        if not is_valid:
            continue
        
        # Extract actual and predicted values
        actual_values = extract_actual_values_from_reference(item)
        predicted_values = extract_predicted_values_from_response(item['response'])
        
        if actual_values is None or predicted_values is None:
            continue
        
        # Evaluate each field
        valid_fields = 0
        for field in NUMERIC_FIELDS:
            if field in actual_values and field in predicted_values:
                error, actual_val, predicted_val = evaluate_numeric_predictions(
                    actual_values[field], predicted_values[field], field
                )
                
                if error is not None:
                    field_errors[field].append(error)
                    field_actuals[field].append(actual_val)
                    field_predictions[field].append(predicted_val)
                    valid_fields += 1
        
        # Consider successful if at least 50% of fields are valid
        if valid_fields >= len(NUMERIC_FIELDS) * 0.5:
            success_count += 1
    
    # Calculate overall metrics
    if not field_errors:
        return None
    
    # Calculate MAE and RMSE for each field
    field_mae = {}
    field_rmse = {}
    
    for field in NUMERIC_FIELDS:
        if field in field_errors and field_errors[field]:
            errors = field_errors[field]
            field_mae[field] = np.mean(errors)
            field_rmse[field] = np.sqrt(np.mean(np.array(errors) ** 2))
    
    # Calculate overall MAE and RMSE
    all_errors = []
    for errors in field_errors.values():
        all_errors.extend(errors)
    
    if all_errors:
        overall_mae = np.mean(all_errors)
        overall_rmse = np.sqrt(np.mean(np.array(all_errors) ** 2))
        success_rate = (success_count / total_count) * 100
    else:
        overall_mae = float('inf')
        overall_rmse = float('inf')
        success_rate = 0
    
    return {
        'Model Name': model_name,
        'MAE': overall_mae,
        'RMSE': overall_rmse,
        'Success Rate (%)': success_rate,
        'Total Samples': total_count,
        'Successful Samples': success_count,
        'Field MAE': field_mae,
        'Field RMSE': field_rmse,
        'Notes': 'Normal Model'
    }

def evaluate_all_models(results_dir, output_csv=None, detailed_dir=None):
    """Evaluate all models in the results directory"""
    if output_csv is None:
        output_csv = 'evaluation_results/comprehensive_model_ranking.csv'
    
    if detailed_dir is None:
        detailed_dir = 'evaluation_results/detailed_reports'
    
    # Find all JSONL files
    jsonl_files = glob.glob(os.path.join(results_dir, "*.jsonl"))
    if not jsonl_files:
        logger.error(f"No JSONL files found in directory {results_dir}")
        return None
    
    logger.info(f"Found {len(jsonl_files)} model result files")
    
    # Evaluate each model
    all_results = []
    
    for file_path in tqdm(jsonl_files, desc="Evaluating models"):
        result = evaluate_single_model(file_path)
        if result is not None:
            all_results.append(result)
    
    if not all_results:
        logger.error("No valid evaluation results found")
        return None
    
    # Create summary DataFrame
    summary_data = []
    for result in all_results:
        summary_data.append({
            'Model Name': result['Model Name'],
            'MAE': result['MAE'],
            'RMSE': result['RMSE'],
            'Success Rate (%)': result['Success Rate (%)'],
            'Total Samples': result['Total Samples'],
            'Successful Samples': result['Successful Samples'],
            'Notes': result['Notes']
        })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary = df_summary.sort_values('MAE')
    
    # Save results
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_summary.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    logger.info("Model Comprehensive Performance Evaluation Results:")
    logger.info(df_summary.to_string(index=False))
    
    # Overall statistics
    total_samples = sum(result['Total Samples'] for result in all_results)
    total_success = sum(result['Successful Samples'] for result in all_results)
    overall_success_rate = (total_success / total_samples) * 100 if total_samples > 0 else 0
    
    logger.info("Overall Statistics:")
    logger.info(f"Total Samples: {total_samples}")
    logger.info(f"Total Success Count: {total_success}")
    logger.info(f"Total Failure Count: {total_samples - total_success}")
    logger.info(f"Overall Success Rate: {overall_success_rate:.2f}%")
    
    # Generate detailed reports if requested
    if detailed_dir:
        generate_detailed_reports(all_results, detailed_dir)
    
    logger.info(f"Comprehensive Evaluation Results Saved to: {output_csv}")
    
    return df_summary

def generate_detailed_reports(all_results, output_dir):
    """Generate detailed reports for each model"""
    os.makedirs(output_dir, exist_ok=True)
    
    for result in all_results:
        model_name = result['Model Name']
        
        # Create detailed report for this model
        report_data = []
        for field in NUMERIC_FIELDS:
            if field in result['Field MAE']:
                report_data.append({
                    'Field': field,
                    'MAE': result['Field MAE'][field],
                    'RMSE': result['Field RMSE'][field]
                })
        
        if report_data:
            df_report = pd.DataFrame(report_data)
            report_file = os.path.join(output_dir, f"{model_name}_detailed_report.csv")
            df_report.to_csv(report_file, index=False, encoding='utf-8-sig')
    
    logger.info(f"Detailed Reports Saved to Directory: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Model Performance Evaluation')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory containing model result files')
    parser.add_argument('--output_csv', type=str, default=None, help='Output CSV file path')
    parser.add_argument('--detailed_dir', type=str, default=None, help='Directory for detailed reports')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs('evaluation_results', exist_ok=True)
    
    # Run evaluation
    results_df = evaluate_all_models(args.results_dir, args.output_csv, args.detailed_dir)
    
    if results_df is not None:
        logger.info("Comprehensive evaluation completed successfully!")
        logger.info(f"Total models evaluated: {len(results_df)}")
        logger.info(f"Results saved to: {args.output_csv}")
    else:
        logger.error("Evaluation failed")

if __name__ == "__main__":
    main() 