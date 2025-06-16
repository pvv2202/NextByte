from ..model_code.init_model import init_next_byte
from flask import Blueprint, jsonify, request

gen = Blueprint('generate', __name__, url_prefix='/api/generate-recipe')


# get our model up in here
model = init_next_byte()

# post requests are usually used to submit new data to be stored in a db
@gen.route('/', methods=['POST'])
def generate_recipe():
    # extracts the json from the body of the request
    data = request.get_json()
    recipe_title = data.get('recipeTitle')
    title, ingredients, directions = model.generate_recipe(
        input_text=f"<start_title>{recipe_title}",
        max_new_tokens=500,
        top_k=10,
        context_length=768
    )
    

    # we return it as a dictionary as well
    return {'recipe': {
        'title': title,
        'ingredients': ingredients,
        'directions': directions
    }}