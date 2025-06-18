#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import glob
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import re
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InstructionFollowingAnalyzer:
    def __init__(self, results_dir="results"):
        self.results_dir = results_dir
        
        # Required 19 flight parameter fields
        self.required_fields = [
            "Latitude (WGS84 deg)", "Longitude (WGS84 deg)", "GPS Altitude (WGS84 ft)",
            "GPS Ground Track (deg true)", "Magnetic Heading (deg)", 
            "GPS Velocity E (m/s)", "GPS Velocity N (m/s)", "GPS Velocity U (m/s)",
            "GPS Ground Speed (kt)", "Roll (deg)", "Pitch (deg)", "Turn Rate (deg/sec)",
            "Slip/Skid", "Normal Acceleration (G)", "Lateral Acceleration (G)",
            "Vertical Speed (fpm)", "Indicated Airspeed (kt)", 
            "Baro Altitude (ft)", "Pressure Altitude (ft)"
        ]
        
    def load_jsonl(self, file_path):
        """Load JSONL file"""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            return data
        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {str(e)}")
            return []
    
    def is_api_error(self, response):
        """Check if response is an API error (call error, not instruction non-compliance)"""
        if not isinstance(response, str):
            return False
        
        response_lower = response.lower()
        error_keywords = [
            'api', 'timeout', 'error', '403', '500', '429', '404',
            'forbidden', 'access denied', 'unauthorized', 'time out',
            'internal server error', 'rate limit', 'not found',
            'connection', 'connect', 'network', 'failed to connect',
            'service unavailable', 'bad request', 'invalid request'
        ]
        
        return any(keyword in response_lower for keyword in error_keywords)
    
    def extract_json_from_response(self, response):
        """Extract JSON content from response"""
        if isinstance(response, dict):
            return response
        
        if not isinstance(response, str):
            return None
        
        # Try direct parsing
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from code blocks
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        matches = re.findall(json_pattern, response, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # Try to find JSON objects
        json_pattern2 = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern2, response, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        return None
    
    def analyze_instruction_following(self, response):
        """Analyze instruction following: API errors are not considered instruction non-compliance"""
        
        # Initialize analysis results
        analysis = {
            'category': '',
            'total_fields': len(self.required_fields),
            'provided_fields': 0,
            'missing_fields': 0,
            'invalid_values': 0,
            'field_completeness': 0.0,
            'missing_field_list': [],
            'invalid_value_list': [],
            'details': '',
            'is_api_error': False,
            'is_valid_attempt': False  # Whether it's a valid instruction following attempt
        }
        
        # 1. API error check (call error, not instruction non-compliance)
        if self.is_api_error(response):
            analysis.update({
                'category': 'API Call Error',
                'details': 'Technical failure causing API error, not instruction non-compliance',
                'is_api_error': True,
                'is_valid_attempt': False
            })
            return analysis
        
        # 2. Response format error check (this is instruction non-compliance)
        if not isinstance(response, str):
            analysis.update({
                'category': 'Response Format Error',
                'details': 'Response is not in string format',
                'is_valid_attempt': True
            })
            return analysis
        
        response = response.strip()
        if not response:
            analysis.update({
                'category': 'Response Format Error',
                'details': 'Empty response',
                'is_valid_attempt': True
            })
            return analysis
        
        # 3. JSON extraction
        json_data = self.extract_json_from_response(response)
        if json_data is None:
            analysis.update({
                'category': 'Response Format Error',
                'details': 'Cannot extract valid JSON format from response',
                'is_valid_attempt': True
            })
            return analysis
        
        # 4. Field completeness and value validity analysis
        missing_fields = []
        invalid_values = []
        provided_fields = 0
        
        analysis['is_valid_attempt'] = True  # Reaching here indicates a valid instruction following attempt
        
        for field in self.required_fields:
            if field not in json_data:
                missing_fields.append(field)
            else:
                provided_fields += 1
                try:
                    value = json_data[field]
                    
                    # Check if value is invalid
                    if value is None:
                        invalid_values.append(f"{field}: null value")
                    elif isinstance(value, str):
                        if value.strip() == "" or value.lower() in ['null', 'none', 'nan', 'n/a', 'undefined']:
                            invalid_values.append(f"{field}: empty or invalid identifier")
                        else:
                            # Try to convert to numeric
                            try:
                                numeric_value = float(value)
                                if math.isnan(numeric_value) or math.isinf(numeric_value):
                                    invalid_values.append(f"{field}: NaN or Inf")
                            except (ValueError, TypeError):
                                invalid_values.append(f"{field}: cannot convert to numeric")
                    elif isinstance(value, (int, float)):
                        if math.isnan(value) or math.isinf(value):
                            invalid_values.append(f"{field}: NaN or Inf")
                    else:
                        invalid_values.append(f"{field}: non-numeric type")
                        
                except Exception as e:
                    invalid_values.append(f"{field}: processing exception ({str(e)})")
        
        # 5. Update analysis results
        analysis.update({
            'provided_fields': provided_fields,
            'missing_fields': len(missing_fields),
            'invalid_values': len(invalid_values),
            'field_completeness': provided_fields / len(self.required_fields),
            'missing_field_list': missing_fields,
            'invalid_value_list': invalid_values
        })
        
        # 6. Determine category
        if len(missing_fields) == 0 and len(invalid_values) == 0:
            analysis.update({
                'category': 'Perfect Response',
                'details': 'All required fields present with valid values'
            })
        elif len(missing_fields) > 0:
            analysis.update({
                'category': 'Missing Fields',
                'details': f'Missing {len(missing_fields)} required fields'
            })
        else:
            analysis.update({
                'category': 'Invalid Values',
                'details': f'Invalid values in {len(invalid_values)} fields'
            })
        
        return analysis
    
    def analyze_all_models(self):
        """Analyze instruction following for all models"""
        logger.info("=" * 80)
        logger.info("Instruction Following Analysis (API errors counted separately)")
        logger.info("=" * 80)
        
        # Find all JSONL files
        jsonl_files = glob.glob(os.path.join(self.results_dir, "*.jsonl"))
        if not jsonl_files:
            logger.error(f"No JSONL files found in {self.results_dir}")
            return None
        
        logger.info(f"Found {len(jsonl_files)} model result files")
        
        all_results = []
        
        for file_path in tqdm(jsonl_files, desc="Analyzing models"):
            model_name = os.path.basename(file_path).replace('.jsonl', '')
            data = self.load_jsonl(file_path)
            
            if not data:
                continue
            
            # Analyze each sample
            total_samples = len(data)
            api_errors = 0
            format_errors = 0
            perfect_responses = 0
            total_completeness = 0
            total_missing_fields = 0
            total_invalid_values = 0
            valid_attempts = 0  # Valid instruction following attempts
            
            for item in data:
                response = item.get('response', '')
                analysis = self.analyze_instruction_following(response)
                
                if analysis['is_api_error']:
                    api_errors += 1
                elif analysis['category'] == 'Response Format Error':
                    format_errors += 1
                elif analysis['category'] == 'Perfect Response':
                    perfect_responses += 1
                
                if analysis['is_valid_attempt']:
                    valid_attempts += 1
                    total_completeness += analysis['field_completeness']
                    total_missing_fields += analysis['missing_fields']
                    total_invalid_values += analysis['invalid_values']
            
            # Calculate metrics
            if valid_attempts > 0:
                avg_completeness = total_completeness / valid_attempts
                avg_missing_per_sample = total_missing_fields / valid_attempts
                avg_invalid_per_sample = total_invalid_values / valid_attempts
            else:
                avg_completeness = 0
                avg_missing_per_sample = 0
                avg_invalid_per_sample = 0
            
            # Calculate rates
            api_error_rate = (api_errors / total_samples) * 100
            format_error_rate = (format_errors / total_samples) * 100
            success_rate = ((total_samples - api_errors - format_errors) / total_samples) * 100
            
            if valid_attempts > 0:
                instruction_compliance_rate = (perfect_responses / valid_attempts) * 100
            else:
                instruction_compliance_rate = 0
            
            # Calculate field problem rate (missing + invalid)
            instruction_issues = total_missing_fields + total_invalid_values
            
            result = {
                'Model Name': model_name,
                'Total Samples': total_samples,
                'API Errors': api_errors,
                'API Error Rate (%)': f"{api_error_rate:.2f}",
                'Format Errors': format_errors,
                'Format Error Rate (%)': f"{format_error_rate:.2f}",
                'Successful Responses': total_samples - api_errors - format_errors,
                'Success Rate (%)': f"{success_rate:.2f}",
                'Valid Attempts': valid_attempts,
                'Valid Attempt Rate (%)': f"{(valid_attempts/total_samples)*100:.2f}",
                'Perfect Responses': perfect_responses,
                'Instruction Compliance Rate (%)': f"{instruction_compliance_rate:.2f}",
                'Average Field Completeness (%)': f"{avg_completeness*100:.2f}",
                'Field Problem Rate (%)': f"{(instruction_issues/valid_attempts)*100:.2f}" if valid_attempts > 0 else "0.00",
                'Average Missing Fields per Sample': f"{avg_missing_per_sample:.2f}",
                'Average Invalid Values per Sample': f"{avg_invalid_per_sample:.2f}"
            }
            
            all_results.append(result)
        
        # Create DataFrame and sort by instruction compliance rate
        df = pd.DataFrame(all_results)
        df = df.sort_values('Instruction Compliance Rate (%)', key=lambda x: x.str.rstrip('%').astype(float), ascending=False)
        
        # Save results
        output_file = 'evaluation_results/instruction_following_analysis.csv'
        os.makedirs('evaluation_results', exist_ok=True)
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        logger.info(f"Instruction following analysis saved to: {output_file}")
        
        # Display top 15 results
        logger.info("Instruction Following Rankings (Top 15):")
        top15 = df.head(15)
        
        for i, (_, row) in enumerate(top15.iterrows(), 1):
            logger.info(f"{i:2d}. {row['Model Name']}")
            logger.info(f"    Instruction Compliance: {row['Instruction Compliance Rate (%)']} | Field Completeness: {row['Average Field Completeness (%)']}")
            logger.info(f"    API Errors: {row['API Error Rate (%)']} | Format Errors: {row['Format Error Rate (%)']} | Valid Attempts: {row['Valid Attempt Rate (%)']}")
            logger.info(f"    Successful Responses: {row['Successful Responses']}/{row['Total Samples']} ({row['Success Rate (%)']})")
            logger.info("")
        
        logger.info("Analysis Notes:")
        logger.info("• API Call Errors: Technical failures, not instruction non-compliance issues")
        logger.info("• Response Format Errors: Cannot extract JSON, belongs to instruction non-compliance")
        logger.info("• Instruction Compliance Rate: Perfect responses / Valid attempts")
        logger.info("• Valid Attempts: Attempts excluding API errors")
        logger.info("• Successful Responses: Total samples - API errors - Format errors")
        logger.info("• Field Problems: Field incompleteness, invalid values, etc.")
        
        return df

def main():
    # Create output directory if it doesn't exist
    os.makedirs('evaluation_results', exist_ok=True)
    
    analyzer = InstructionFollowingAnalyzer()
    results = analyzer.analyze_all_models()
    
    if results is not None:
        logger.info("Instruction following analysis completed successfully!")
    else:
        logger.error("Analysis failed")

if __name__ == "__main__":
    main() 