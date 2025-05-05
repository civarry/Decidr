from flask import Flask

# Import blueprints
from routes.decision_routes import decision_bp
from routes.prediction_routes import prediction_bp

def register_blueprints(app: Flask):
    """Register all blueprints with the app."""
    app.register_blueprint(decision_bp)
    app.register_blueprint(prediction_bp)