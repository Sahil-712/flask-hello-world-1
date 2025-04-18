from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/about')
def about():
    return 'About'

@app.route('/square', methods=['POST'])
def square():
    data = request.get_json()
    number = data.get('number')
    if number is None:
        return {'error': 'No number provided'}, 400
    try:
        number = float(number)
        return {'square': number ** 2}
    except ValueError:
        return {'error': 'Invalid number'}, 400


