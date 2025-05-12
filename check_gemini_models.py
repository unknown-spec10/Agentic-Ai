import google.generativeai as genai
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use the API key from your secrets.toml file
API_KEY = "AIzaSyDwBUPK4SjQs8cq7-agMJPSWOYhIRpUKBY"

def check_available_models():
    """Check which models are available for the current API key and what methods they support."""
    try:
        # Configure the Gemini API with your key
        genai.configure(api_key=API_KEY)
        
        # Get the list of available models
        print("Listing available models...")
        available_models = list(genai.list_models())
        
        if not available_models:
            print("No models were returned. This might indicate an issue with the API key permissions.")
            return
            
        print(f"Found {len(available_models)} models.")
        
        # Print details for each model
        for i, model in enumerate(available_models):
            print(f"\n--- Model {i+1} ---")
            print(f"Name: {model.name}")
            
            if hasattr(model, 'display_name'):
                print(f"Display Name: {model.display_name}")
                
            if hasattr(model, 'description'):
                print(f"Description: {model.description}")
                
            if hasattr(model, 'supported_generation_methods'):
                print(f"Supported Methods: {', '.join(model.supported_generation_methods)}")
                
            if hasattr(model, 'input_token_limit'):
                print(f"Input Token Limit: {model.input_token_limit}")
                
            if hasattr(model, 'output_token_limit'):
                print(f"Output Token Limit: {model.output_token_limit}")
        
        # Try to test each model
        print("\n\nTesting models for content generation...")
        for model in available_models:
            try:
                if hasattr(model, 'supported_generation_methods') and 'generateContent' in model.supported_generation_methods:
                    print(f"\nTesting model: {model.name}")
                    # Create the model
                    gen_model = genai.GenerativeModel(model.name)
                    # Simple test prompt
                    response = gen_model.generate_content("Hello, how are you?")
                    print(f"✅ Success! Response: {response.text[:50]}...")
                else:
                    print(f"\nSkipping model {model.name} (doesn't support generateContent)")
            except Exception as e:
                print(f"❌ Error testing {model.name}: {str(e)}")
                
    except Exception as e:
        print(f"Failed to list models: {str(e)}")

if __name__ == "__main__":
    check_available_models()
