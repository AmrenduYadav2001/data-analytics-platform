from app import create_app
import os

app = create_app()

if __name__ == "__main__":

    debug_mode = os.environ.get("FLASK_DEBUG", "False") == "True"

    app.run(
        debug=True
    )