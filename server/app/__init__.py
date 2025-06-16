from redis import Redis
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from .database import SqliteNextByteDB

db = SqliteNextByteDB(num_connections=1)

from app.routes.auth import auth
from app.routes.generate import gen

def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)
    
    app.config['SECRET_KEY'] = os.environ.get('SECRETIVE_SECRET')
    app.register_blueprint(auth)
    app.register_blueprint(gen)
    print(app.url_map)
    
    return app