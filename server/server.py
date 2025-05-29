from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
from model_code.init_model import init_next_byte
import re

"""NOT MUCH HERE YET BUT..."""

""" 
flask simplifies setting up api routes, here I just instantiate the app server,
enable cross referencing (allowing the server to admit http requests from other domains/ports),
load the model in, and define the generate recipe route
"""
app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://192.168.2.109:3000"
    "http://10.0.0.84:3000"
], supports_credentials=True)

# get our model up in here
model = init_next_byte()

# post requests are usually used to submit new data to be stored in a db
@app.route('/api/generate-recipe', methods=['POST'])
def generate_recipe():
    # extracts the json from the body of the request
    data = request.get_json()
    recipe_title = data.get('recipeTitle')
    output = model.generate_recipe(
        input_text=recipe_title,
        max_new_tokens=500,
        top_k=10,
        context_length=768
    )
    
    title_end = output.find("<end_title>")
    ingredients_end = output.find("<end_ingredients>")
    directions_end = output.find("<end>")

    title = output[:title_end].strip()
    ingredients = output[title_end + len("<end_title>"):ingredients_end].strip()
    directions = output[ingredients_end + len("<end_ingredients>"):directions_end].strip()

    # Clean up spaces before punctuation
    title = re.sub(r'\s+([.,!?;:])', r'\1', title).strip().capitalize()
    ingredients = re.sub(r'\s+([.,!?;:])', r'\1', ingredients)
    directions = re.sub(r'\s+([.,!?;:])', r'\1', directions)

    # Split ingredients on comma followed by a digit
    ingredients = [i.strip() for i in re.split(r',\s*(?=\d)', ingredients) if i.strip()]
    directions = [s.strip().capitalize() for s in directions.split('.') if s.strip()]
# ...existing code...
    # we return it as a dictionary as well
    return {'recipe': {
        'title': title,
        'ingredients': ingredients,
        'directions': directions
    }}
