#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import logging
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinalModelRanking:
    def __init__(self):
        self.accuracy_file = 'evaluation_results/comprehensive_model_ranking.csv'
        self.instruction_file = 'evaluation_results/instruction_following_analysis.csv'
        
    def load_data(self):
        """Load accuracy and instruction following data"""
        try:
            # Load accuracy data
            self.accuracy_df = pd.read_csv(self.accuracy_file)
            logger.info(f"Loaded accuracy data: {len(self.accuracy_df)} models")
            
            # Load instruction following data
            self.instruction_df = pd.read_csv(self.instruction_file)
            logger.info(f"Loaded instruction following data: {len(self.instruction_df)} models")
            
            return True
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            return False
    
    def calculate_absolute_accuracy_score(self, mae, rmse):
        """Calculate absolute accuracy score (0-100 points)"""
        # MAE scoring (60% weight)
        if mae < 5:
            mae_score = 100 - (mae / 5) * 10  # 90-100 points
        elif mae < 15:
            mae_score = 90 - ((mae - 5) / 10) * 20  # 70-90 points
        elif mae < 50:
            mae_score = 70 - ((mae - 15) / 35) * 30  # 40-70 points
        else:
            mae_score = max(0, 40 - ((mae - 50) / 50) * 40)  # 0-40 points
        
        # RMSE scoring (40% weight)
        if rmse < 10:
            rmse_score = 100 - (rmse / 10) * 10  # 90-100 points
        elif rmse < 30:
            rmse_score = 90 - ((rmse - 10) / 20) * 20  # 70-90 points
        elif rmse < 100:
            rmse_score = 70 - ((rmse - 30) / 70) * 30  # 40-70 points
        else:
            rmse_score = max(0, 40 - ((rmse - 100) / 100) * 40)  # 0-40 points
        
        # Combined score
        accuracy_score = mae_score * 0.6 + rmse_score * 0.4
        return min(100, max(0, accuracy_score))
    
    def calculate_instruction_following_score(self, compliance_rate, field_completeness, field_problem_rate, format_error_rate):
        """Calculate instruction following score (0-100 points) - only considers field completeness, field validity, and format correctness"""
        # Field completeness scoring (50% weight)
        if isinstance(field_completeness, str):
            completeness_score = float(field_completeness.rstrip('%'))
        else:
            completeness_score = float(field_completeness)
        
        # Field validity scoring (30% weight) - lower field problem rate is better
        if isinstance(field_problem_rate, str):
            problem_penalty = float(field_problem_rate.rstrip('%'))
        else:
            problem_penalty = float(field_problem_rate)
        validity_score = max(0, 100 - problem_penalty)
        
        # Format correctness scoring (20% weight) - lower format error rate is better
        if isinstance(format_error_rate, str):
            format_penalty = float(format_error_rate.rstrip('%'))
        else:
            format_penalty = float(format_error_rate)
        format_score = max(0, 100 - format_penalty * 2)  # Format error penalty
        
        # Combined score
        instruction_score = (
            completeness_score * 0.5 +
            validity_score * 0.3 +
            format_score * 0.2
        )
        
        return min(100, max(0, instruction_score))
    
    def merge_and_calculate_scores(self):
        """Merge data and calculate final model scores"""
        logger.info("=" * 80)
        logger.info("Final Model Capability Scoring (excluding system reliability)")
        logger.info("=" * 80)
        
        # Merge data
        merged_data = []
        
        for _, acc_row in self.accuracy_df.iterrows():
            model_name = acc_row['Model Name']
            
            # Find corresponding instruction following data
            inst_row = self.instruction_df[self.instruction_df['Model Name'] == model_name]
            
            if len(inst_row) == 0:
                logger.warning(f"Instruction following data not found for {model_name}")
                continue
            
            inst_row = inst_row.iloc[0]
            
            # Extract data
            mae = float(acc_row['MAE'])
            rmse = float(acc_row['RMSE'])
            
            # Calculate scores
            accuracy_score = self.calculate_absolute_accuracy_score(mae, rmse)
            instruction_score = self.calculate_instruction_following_score(
                inst_row['Instruction Compliance Rate (%)'],
                inst_row['Average Field Completeness (%)'],
                inst_row['Field Problem Rate (%)'],
                inst_row['Format Error Rate (%)']
            )
            
            # Final model score (accuracy 60% + instruction following 40%)
            final_score = (
                accuracy_score * 0.6 +
                instruction_score * 0.4
            )
            
            # Determine model grade
            if final_score >= 90:
                model_grade = "S Grade"
            elif final_score >= 80:
                model_grade = "A Grade"
            elif final_score >= 70:
                model_grade = "B Grade"
            elif final_score >= 60:
                model_grade = "C Grade"
            else:
                model_grade = "D Grade"
            
            # Calculate API reliability (for reference only, not included in total score)
            api_error_rate = float(inst_row['API Error Rate (%)']) if isinstance(inst_row['API Error Rate (%)'], str) else inst_row['API Error Rate (%)']
            api_reliability = max(0, 100 - api_error_rate)
            
            merged_data.append({
                'Rank': 0,  # Fill later
                'Model Name': model_name,
                'Model Grade': model_grade,
                'Overall Score': f"{final_score:.2f}",
                'Accuracy Score': f"{accuracy_score:.2f}",
                'Instruction Following Score': f"{instruction_score:.2f}",
                'MAE': f"{mae:.4f}",
                'RMSE': f"{rmse:.4f}",
                'Instruction Compliance Rate (%)': inst_row['Instruction Compliance Rate (%)'],
                'Field Completeness (%)': inst_row['Average Field Completeness (%)'],
                'Format Error Rate (%)': inst_row['Format Error Rate (%)'],
                'Field Problem Rate (%)': inst_row['Field Problem Rate (%)'],
                'API Reliability (%)': f"{api_reliability:.2f}",
                'API Error Rate (%)': inst_row['API Error Rate (%)'],
                'Valid Attempts': inst_row['Valid Attempts'],
                'Successful Responses': inst_row['Successful Responses'],
                'Total Samples': inst_row['Total Samples'],
                'Notes': acc_row['Notes'] if acc_row['Notes'] != 'Normal Model' else ''
            })
        
        # Sort by overall score
        merged_data.sort(key=lambda x: float(x['Overall Score']), reverse=True)
        
        # Assign rankings
        for i, item in enumerate(merged_data, 1):
            item['Rank'] = i
        
        # Create DataFrame
        self.final_df = pd.DataFrame(merged_data)
        
        return self.final_df
    
    def generate_final_report(self):
        """Generate final model evaluation report"""
        # Save results
        output_file = 'evaluation_results/final_model_ranking.csv'
        self.final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        logger.info(f"Final model rankings saved to: {output_file}")
        
        # Display top 15
        logger.info("Final Model Capability Rankings (Top 15):")
        top15 = self.final_df.head(15)
        
        for _, row in top15.iterrows():
            logger.info(f"{row['Rank']:2d}. {row['Model Name']} ({row['Model Grade']})")
            logger.info(f"    Overall Score: {row['Overall Score']} | Accuracy: {row['Accuracy Score']} | Instruction Following: {row['Instruction Following Score']}")
            logger.info(f"    MAE: {row['MAE']} | RMSE: {row['RMSE']} | Instruction Compliance: {row['Instruction Compliance Rate (%)']}%")
            logger.info(f"    Format Errors: {row['Format Error Rate (%)']}% | Field Problems: {row['Field Problem Rate (%)']}% | API Reliability: {row['API Reliability (%)']}% (reference only)")
            if row['Notes']:
                logger.info(f"    Notes: {row['Notes']}")
            logger.info("")
        
        # Statistical analysis
        grade_counts = self.final_df['Model Grade'].value_counts()
        logger.info("Grade Distribution Statistics:")
        for grade, count in grade_counts.items():
            logger.info(f"  {grade}: {count} models")
        
        # Best models in each dimension
        best_accuracy = self.final_df.loc[self.final_df['Accuracy Score'].astype(float).idxmax()]
        best_instruction = self.final_df.loc[self.final_df['Instruction Following Score'].astype(float).idxmax()]
        best_reliability = self.final_df.loc[self.final_df['API Reliability (%)'].astype(float).idxmax()]
        
        logger.info("Best Models in Each Dimension:")
        logger.info(f"Best Accuracy: {best_accuracy['Model Name']} ({best_accuracy['Accuracy Score']} points)")
        logger.info(f"Best Instruction Following: {best_instruction['Model Name']} ({best_instruction['Instruction Following Score']} points)")
        logger.info(f"Best API Reliability: {best_reliability['Model Name']} ({best_reliability['API Reliability (%)']}%) (reference only)")
        
        # Production recommendations
        logger.info("Production Environment Recommendations:")
        recommended_models = self.final_df[self.final_df['Model Grade'].isin(['S Grade', 'A Grade'])]
        
        logger.info("Recommended Models (S Grade/A Grade):")
        for _, model in recommended_models.iterrows():
            reliability_status = "High" if float(model['API Reliability (%)']) >= 95 else "Medium" if float(model['API Reliability (%)']) >= 80 else "Low"
            logger.info(f"  • {model['Model Name']} ({model['Model Grade']}, Overall: {model['Overall Score']}, API Reliability: {reliability_status})")
        
        logger.info("Final Scoring Criteria:")
        logger.info("• Overall Score = Accuracy Score 60% + Instruction Following Score 40%")
        logger.info("• Accuracy Score: Based on MAE and RMSE absolute scoring")
        logger.info("• Instruction Following Score: Field Completeness (50%) + Field Validity (30%) + Format Correctness (20%)")
        logger.info("• API Reliability: For reference only, not included in model scoring (platform issue)")
        logger.info("• Format Errors: JSON parsing errors, structural issues, etc.")
        logger.info("• Field Problems: Invalid field values, data type errors, etc.")
        logger.info("• Model Grades: S Grade (≥90), A Grade (80-90), B Grade (70-80), C Grade (60-70), D Grade (<60)")
        
        return self.final_df
    
    def run_analysis(self):
        """Run the complete analysis"""
        if not self.load_data():
            return None
        
        results = self.merge_and_calculate_scores()
        self.generate_final_report()
        
        logger.info("Final model capability scoring analysis completed!")
        logger.info(f"Total models evaluated: {len(results)}")
        logger.info("Final version features:")
        logger.info("• Only evaluates model capability: accuracy + instruction following")
        logger.info("• API reliability for reference only, not included in model scoring")
        logger.info("• Weights: accuracy 60% + instruction following 40%")
        logger.info("• Instruction following includes: compliance rate, field completeness, field problems, format correctness")
        logger.info("• Provides pure model capability rankings for production environments")
        
        return results

def main():
    # Create output directory if it doesn't exist
    import os
    os.makedirs('evaluation_results', exist_ok=True)
    
    ranking = FinalModelRanking()
    results = ranking.run_analysis()
    
    if results is not None:
        logger.info("Final ranking analysis completed successfully!")
    else:
        logger.error("Analysis failed")

if __name__ == "__main__":
    main() 