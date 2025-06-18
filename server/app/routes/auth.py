import json
from flask import Blueprint, jsonify, request, session
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import datetime
from .. import db


auth = Blueprint('auth', __name__, url_prefix='/api/auth')


@auth.route('/@me', methods=['GET'])
def get_user():
    print('getting user info')
    if 'user_id' not in session:
        return jsonify({'error': 'not logged in'}), 401
    user = db.execute_query(f'SELECT * FROM Users WHERE id = ?', (session['user_id'],))
    
    return quick_user_info(user), 200
    
    

@auth.route('/login', methods=['POST'])
def login():
    print('login route')
    if 'user_id' in session:
        return jsonify({'error': 'already logged in'}), 409
    user_data = request.get_json()
    username = user_data['username']
    password = user_data['password']
    # find the user if present in db
    user = db.execute_query(f'SELECT * FROM Users WHERE username = ?', (username,))
    if not user:
        return jsonify({'error': 'User not found'}), 404

    # TODO: change with hashed password after creating signup
   
    if not check_password_hash(user[2], password):
        return jsonify({'error' : f'Incorrect Password'}), 401
    
    session['user_id'] = user[0]
    
    return quick_user_info(user), 200
    
@auth.route('/signup', methods=['POST'])
def signup():
    print('in signup')
    data = request.get_json()
    username, age, email = data['username'], data['age'], data['email']
    city, state, country = data['city'], data['state'], data['country']
    date_created = datetime.now().strftime('%Y-%m-%d')
    # encrypt password
    password = generate_password_hash(data['password'])
    
    if db.execute_query(f'SELECT * FROM Users WHERE username = ?', (username,)):
        # 409 for conflict 
        return jsonify({'error': 'Username is taken'}), 409
    
   
    query = f"""
    INSERT INTO Users
    (username, password, age, email, date_created, city, state, country)
    VALUES(?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    db.insert(query, (username, password, age, email, date_created, city, state, country))
    
    
    return jsonify({'msg': 'OK'}), 201

@auth.route('/signout', methods=['POST'])
def signout():
    print('signing out')
    if 'user_id' not in session:
        return jsonify({'error': 'not signed in'}), 401
    
    session.pop('user_id')
    
    return jsonify({'msg': 'Signed out'}), 200


def quick_user_info(user):
    
    return jsonify({
        'username': user[1],
        'email': user[3],
        'country': user[4],
        'state': user[5],
        'city': user[6],
        'age': user[7],
        'date_created': user[8]
    })
    