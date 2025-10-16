from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    """Render the home page"""
    try:
        return render_template('index.html')
    except Exception as e:
        return f"""
        <html>
        <head><title>Error</title></head>
        <body>
            <h1>Error Loading Template</h1>
            <p>{str(e)}</p>
        </body>
        </html>
        """

@app.route('/predict', methods=['POST'])
def predict():
    """Simple prediction endpoint (no actual model)"""
    try:
        # Get form data
        airline = request.form.get('airline')
        origin = request.form.get('origin')
        dest = request.form.get('dest')
        
        # For testing - just return a simple response
        return f"""
        <html>
        <head><title>Prediction Result</title></head>
        <body>
            <h1>Prediction Result</h1>
            <p>Airline: {airline}</p>
            <p>Origin: {origin}</p>
            <p>Destination: {dest}</p>
            <p><b>Result: Flight is predicted to be on time.</b></p>
            <p><a href="/">Back to form</a></p>
        </body>
        </html>
        """
    except Exception as e:
        return f"""
        <html>
        <head><title>Error</title></head>
        <body>
            <h1>Error Processing Request</h1>
            <p>{str(e)}</p>
        </body>
        </html>
        """

if __name__ == '__main__':
    app.run(debug=True)