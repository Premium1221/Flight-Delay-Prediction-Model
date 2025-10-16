#!/usr/bin/env python
"""
Flight Delay Predictor - Application Starter
Run this script to start the Flask application
"""

import os
from app import app

if __name__ == "__main__":
    # Set environment variables
    os.environ.setdefault('FLASK_APP', 'app.py')
    
    # Determine if we're in development or production
    debug_mode = True
    
    # Print startup message
    host = '0.0.0.0'
    port = 5000
    print("=" * 70)
    print(f"Flight Delay Predictor is starting up!")
    print(f"Running on http://127.0.0.1:{port} (Press CTRL+C to quit)")
    print("=" * 70)
    
    # Run the Flask application
    app.run(debug=debug_mode, host=host, port=port)