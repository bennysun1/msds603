import requests
import json
import sys
import time

def test_root_endpoint():
    """Test the root endpoint of the API"""
    try:
        url = 'http://127.0.0.1:8000/'
        response = requests.get(url)
        print("Root Endpoint Response:")
        print(response.json())
        print("Status Code:", response.status_code)
        print("-" * 50)
        return True
    except Exception as e:
        print(f"Error testing root endpoint: {e}")
        return False

def test_prediction_endpoint():
    """Test the prediction endpoint with different comments"""
    
    url = 'http://127.0.0.1:8000/predict'
    
    # Test cases
    test_comments = [
        "This is a helpful comment that should stay.",
        "Useless comment, you should flag it for removal",
        "I love this post, very informative!",
        "This adds nothing to the conversation and should be removed.",
        "Check out my website at www.spam.com to buy cheap products!"
    ]
    
    all_successful = True
    
    for i, comment in enumerate(test_comments):
        try:
            data = {'reddit_comment': comment}
            response = requests.post(url, json=data)
            
            print(f"Test Case {i+1}: '{comment}'")
            print("Response:", response.json())
            print("Status Code:", response.status_code)
            
            if response.status_code != 200:
                all_successful = False
                
        except Exception as e:
            print(f"Error in test case {i+1}: {e}")
            all_successful = False
            
        print("-" * 50)
    
    return all_successful

if __name__ == "__main__":
    print("Testing Reddit Comment Classifier API")
    print("=" * 50)
    
    # Test root endpoint
    root_success = test_root_endpoint()
    
    # Wait a moment to ensure model is loaded
    print("Waiting for model to load...")
    time.sleep(2)
    
    # Test prediction endpoint
    prediction_success = test_prediction_endpoint()
    
    # Report overall success
    if root_success and prediction_success:
        print("All tests completed successfully!")
        sys.exit(0)
    else:
        print("Some tests failed. Check the output above for details.")
        sys.exit(1) 