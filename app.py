from flask import Flask, request, jsonify, render_template
from woa_algorithm import WOA

app = Flask(__name__)

@app.route('/')
def home():
    # Serve the frontend HTML file
    return render_template('app.html')

@app.route('/optimize', methods=['POST'])
def optimize():
    """
    API endpoint to run Whale Optimization Algorithm.
    Expects JSON input with DG locations, population size, iterations, and bounds.
    """
    data = request.json
    if not data or 'resistances' not in data:
        return jsonify({'error': 'Invalid input, expected JSON with "resistances" key'}), 400


    best_solution, min_loss = WOA(data)
    return jsonify({
        'best_solution': best_solution.tolist(),
        'minimum_loss': min_loss
    })

if __name__ == '__main__':
    app.run(debug=True)

