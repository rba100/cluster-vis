import os
from flask import Flask, request, jsonify
from api_getthemes import getThemes

app = Flask(__name__)

@app.route('/text-analysis/themes', methods=['POST'])
def get_themes():
    query_params = request.args.to_dict()
    request_body = request.get_json()

    data = request_body.get('data', None)

    k = query_params.get('k', None)
    if k is not None and not k.isdigit():
        return jsonify({ "error": "k must be a positive integer" }), 400    
    k = int(k) if k is not None else None

    commonConcept = query_params.get('commonConcept', None)

    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        return jsonify({ "error": "data must be a list of strings" }), 400
    
    response_data = getThemes(data, k, commonConcept)

    return response_data

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2222)
