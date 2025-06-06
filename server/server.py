from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
from model_code.init_model import init_next_byte
import re
from db import MySQLNextByteDB
import datetime

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
    "http://192.168.2.109:3000",
    "http://10.0.0.84:3000"
], supports_credentials=True)

# get our model up in here
model = init_next_byte()

# init database/connection pool
db = MySQLNextByteDB()

# post requests are usually used to submit new data to be stored in a db
@app.route('/api/generate-recipe', methods=['POST'])
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
    
@app.route('/api/login', methods=['POST'])
def login():
    print('login route')
    user_data = request.get_json()
    username = user_data['username']
    password = user_data['password']
    # find the user if present in db
    user = db.execute_query(f'SELECT * FROM Users WHERE username = %s', (username,))
    if not user:
        return jsonify({'error': 'User not found'}), 404

    # TODO: change with hashed password after creating signup
    if user['password'] != password:
        return jsonify({'error' : f'Incorrect Password'}), 401
    
    return jsonify({'msg': 'OK'}), 200

@app.route('/api/signup', methods=['POST'])
def signup():
    print('in signup')
    data = request.get_json()
    username, password, email = data['username'], data['password'], data['email']
    city, state, country = data['city'], data['state'], data['country']
    date_created = datetime.now().strftime('%Y-%m-%d')
    
    if db.execute_query(f'SELECT * FROM Users WHERE username = %s', (username,)):
        # 409 for conflict 
        return jsonify({'error': 'Username is taken'}), 409
    
    query = f"""
    INSERT INTO Users
    (username, password, email, date_created, city, state, country)
    VALUES(%s, %s, %s, %s, %s, %s, %s)
    """
    
    db.execute_query(query,(username, password, email, date_created, city, state, country))
    
    return jsonify({'msg': 'OK'}), 201
    
    
   
    

if __name__ == "__main__":
    app.run(debug=True)
        
    
