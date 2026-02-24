
import os
import json
from google import genai
from google.genai import types

# Use user-provided key
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyB-pbjhjTokPj6Od0goDldEtKPdZ2pRfnY")

def test_gemini():
    print(f"Testing Gemini with key ending in ...{GEMINI_API_KEY[-4:]}")
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        # Test Structured Output on 2.5-flash
        print("\nTest 4: Structured output (gemini-2.5-flash)")
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents="Grade this correct code: print('hello'). Return JSON with score 10 and feedback 'Good'.",
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "score": {"type": "NUMBER"},
                            "feedback": {"type": "STRING"}
                        },
                        "required": ["score", "feedback"]
                    }
                )
            )
            print(f"Success! Response: {response.text}")
        except Exception as e:
            print(f"Failed structured output with gemini-2.5-flash: {e}")
            
        # Test Manual JSON
        print("\nTest 5: Manual JSON (gemini-2.5-flash)")
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents="Grade this correct code: print('hello'). Return ONLY valid JSON with keys 'score' (number) and 'feedback' (string). No markdown."
            )
            print(f"Success! Response: {response.text}")
        except Exception as e:
            print(f"Failed manual JSON with gemini-2.5-flash: {e}")

    except Exception as e:
        print(f"Client initialization failed: {e}")

if __name__ == "__main__":
    test_gemini()
