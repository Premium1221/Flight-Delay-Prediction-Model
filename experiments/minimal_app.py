from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Minimal Test</title>
    </head>
    <body>
        <h1>Flight Delay Predictor - Test Page</h1>
        <p>This is a simple test page to verify the Flask server is working correctly.</p>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(debug=True)