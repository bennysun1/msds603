#!/bin/bash

# Display the curl command being executed
echo "Running curl command to test the prediction endpoint:"
echo "curl -X 'POST' 'http://127.0.0.1:8000/predict' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{\"reddit_comment\": \"Useless comment, you should flag it for removal\"}'"
echo ""

# Execute the curl command
echo "Response:"
curl -X 'POST' 'http://127.0.0.1:8000/predict' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"reddit_comment": "Useless comment, you should flag it for removal"}'
echo ""  # Add a newline after the response 