from flask import Flask
from flask_bootstrap import Bootstrap5
from flask_wtf import CSRFProtect

from config import Config
from routes import register_blueprints

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize extensions
    bootstrap = Bootstrap5(app)
    csrf = CSRFProtect(app)
    
    # Register all route blueprints
    register_blueprints(app)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)