from redis import Redis
from flask import Flask, request, jsonify
from flask_cors import CORS
from .database import SqliteNextByteDB
from flask_session import Session
from .config import ApplicationConfig

db = SqliteNextByteDB(num_connections=1)

from app.routes.auth import auth
from app.routes.recipes import recipes

def create_app():
    app = Flask(__name__)
    
    CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)
    
    app.config.from_object(ApplicationConfig)
    Session(app)
    
    app.register_blueprint(auth)
    app.register_blueprint(recipes)
   
    
    return app