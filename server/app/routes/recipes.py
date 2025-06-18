from ..model_code.init_model import init_next_byte
from flask import Blueprint, jsonify, request, session
from datetime import datetime
from .. import db
recipes = Blueprint('recipes', __name__, url_prefix='/api/recipes')


# get our model up in here
model = init_next_byte()

# post requests are usually used to submit new data to be stored in a db
@recipes.route('/generate', methods=['POST'])
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

@recipes.route('/saved-recipes', methods=['GET', 'POST'])
def saved_recipes():
    if 'user_id' not in session:
        return jsonify({'error': 'not logged in'}), 401
    
    print('generic saved recipes route')
    user_id = session['user_id']
    
    
    if request.method == 'POST':
        print('posting new recipe')
        data = request.get_json()
        title, ingredients, directions = data['title'], data['ingredients'], data['directions']
        date_created = datetime.now().strftime('%Y-%m-%d')
        
        query = f"""
            INSERT INTO Recipes
            (title, ingredients, directions, date_created, user_id)
            VALUES(?, ?, ?, ?, ?)
            """
    
        db.insert(query, (title, ingredients, directions, date_created, user_id))
    
        return jsonify({'msg': 'OK'}), 201

    if request.method == 'GET':
        query =f"""
                SELECT * FROM Recipes 
                WHERE user_id = ?
                LIMIT 20
                """
        recipes = db.execute_query(query=query, params=(user_id,), many=True)
        
        return recipes, 200
    
    return jsonify({'error': 'error occurred'}), 400
    
   
    
@recipes.route('/saved-recipes/<int:recipe_id>', methods=['GET', 'DELETE', 'PUT', 'PATCH'])
def one_recipe(recipe_id):
    
    # ENSURE USER IS AUTHENTICATED
    if 'user_id' not in session:
        return jsonify({'error': 'not logged in'}), 401
    
    if request.method == 'GET':
        # extract the full recipe info with recipe id
        query = f"SELECT * FROM Recipes WHERE id = ?"
        recipe = db.execute_query(query=query, params=(recipe_id,), many=False)
        
        return jsonify(recipe), 200
    
    elif request.method == 'DELETE':
        # delete the recipe in question from the records
        pass
    elif request.method == 'PUT':
        # fully update fields 
        pass
    elif request.method == 'PATCH':
        # update select fields
        pass
    
    else: 
        return jsonify({'error': 'invalid request method'}), 404
    