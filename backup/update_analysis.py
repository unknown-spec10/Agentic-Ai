# update_analysis.py
import json
import os
import logging
from utils import generate_learning_recommendations, MATCH_THRESHOLD, recommend_job_roles

def generate_revised_skill_gap_analysis(analysis_json_path):
    """
    Revise the skill gap analysis with better matching criteria
    """
    try:
        # Load the existing analysis
        with open(analysis_json_path, 'r', encoding='utf-8') as f:
            analysis = json.load(f)
        
        # Reclassify skills based on distance threshold from utils
        true_matches = []
        actual_missing_skills = []
        
        for match in analysis["skill_gap_analysis"]["matched_skills"]:
            if match["distance"] < MATCH_THRESHOLD:
                true_matches.append(match)
            else:
                actual_missing_skills.append(match["job_skill"])
        
        # Add any already identified missing skills
        for skill in analysis["skill_gap_analysis"]["missing_skills"]:
            if skill not in actual_missing_skills:
                actual_missing_skills.append(skill)
        
        # Update the analysis
        analysis["skill_gap_analysis"]["matched_skills"] = true_matches
        analysis["skill_gap_analysis"]["missing_skills"] = actual_missing_skills
        analysis["skill_gap_analysis"]["match_percentage"] = (len(true_matches) / len(analysis["job_required_skills"])) * 100 if analysis["job_required_skills"] else 0
        
        return analysis
        
    except Exception as e:
        print(f"Error revising skill gap analysis: {e}")
        return None


def update_skill_analysis(analysis_file_path=None):
    """
    Update the existing skill gap analysis with better results
    
    Args:
        analysis_file_path (str): Path to the analysis file to update
    """
    try:
        # Use provided path or default to current directory
        if not analysis_file_path:
            analysis_file_path = os.path.join(os.getcwd(), "skill_gap_analysis.json")
            
        # Check if the file exists
        if not os.path.exists(analysis_file_path):
            print(f"❌ Analysis file not found at {analysis_file_path}")
            return None
            
        # Load existing skill gap analysis
        with open(analysis_file_path, 'r', encoding='utf-8') as f:
            revised_analysis = json.load(f)

        # If job role recommendations don't exist in the original analysis, add them
        if "job_role_recommendations" not in revised_analysis:
            job_role_recommendations = recommend_job_roles(revised_analysis["resume_skills"])
            revised_analysis["job_role_recommendations"] = job_role_recommendations
        
        # Print job role recommendations in summary
        print("\n💼 Recommended Job Roles:")
        for role in revised_analysis.get('job_role_recommendations', [])[:3]:
            print(f"- {role}")

        # Generate learning recommendations
        recommendations = generate_learning_recommendations(revised_analysis["skill_gap_analysis"]["missing_skills"])
        revised_analysis["learning_recommendations"] = recommendations

        # Save updated analysis in the same directory as the input file
        output_path = os.path.join(os.path.dirname(analysis_file_path), "updated_skill_gap_analysis.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(revised_analysis, f, indent=4, ensure_ascii=False)

        print(f"✅ Updated skill gap analysis saved to {output_path}")
        return revised_analysis
    
    except Exception as e:
        print(f"❌ Error updating skill analysis: {e}")

# Run the update function when executed as a script
if __name__ == "__main__":
    try:
        print("🚀 Starting skill gap analysis update...")
        update_skill_analysis()
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        print("✅ Process complete.")
