import os
from flask import Flask, request, jsonify
from api_getthemes import getThemes

app = Flask(__name__)

@app.route('/text-analysis/themes', methods=['POST'])
def get_themes():
    query_params = request.args.to_dict()
    request_body = request.get_json()

    input = request_body.get('input', None)

    k = query_params.get('k', None)
    if (k is not None and k != "auto") and not k.isdigit():
        return jsonify({ "error": "k must be a positive integer or 'auto'" }), 400

    commonConcept = query_params.get('commonConcept', None)

    if not isinstance(input, list) or not all(isinstance(item, str) for item in input):
        return jsonify({ "error": "input must be a list of strings" }), 400
    
    response_data = getThemes(input, k, commonConcept)

    return response_data

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2222)
