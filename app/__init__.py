from flask import Flask
from .extensions import db  # ✅ moved to top
import os


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-fallback-key")
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        "mysql+pymysql://root:Amrendu_Yadav2001@localhost/customer_analysis"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")


def create_app():

    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    from .models.model import User, UploadFile

    with app.app_context():
        db.create_all()

    # blueprints
    from .routes.auth import auth
    app.register_blueprint(auth)

    # ✅ stock blueprint added
    from .routes.stock import stock
    app.register_blueprint(stock)

    return app