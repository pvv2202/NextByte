from flask import Flask, request
from flask_cors import CORS
from pathlib import Path
from model_code.init_model import init_next_byte

"""NOT MUCH HERE YET BUT..."""

""" 
flask simplifies setting up api routes, here I just instantiate the app server,
enable cross referencing (allowing the server to admit http requests from other domains/ports),
load the model in, and define the generate recipe route
"""
app = Flask(__name__)
CORS(app)

# get our model up in here
model = init_next_byte()

# post requests are usually used to submit new data to be stored in a db
@app.route('/api/generate-recipe', methods=['POST'])
def generate_recipe():
    # extracts the json from the body of the request
    data = request.get_json()
    recipe_title = data.get('recipeTitle')
    recipe = model.generate_recipe(
        input_text=recipe_title,
        max_new_tokens=400,
        top_k=10,
        context_length=512
    )
    # we return it as a dictionary as well
    return {'recipe': recipe}