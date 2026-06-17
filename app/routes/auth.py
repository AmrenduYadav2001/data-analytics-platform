from flask import Blueprint, session, url_for, redirect, request, render_template, flash, current_app
from werkzeug.utils import secure_filename
import os
import uuid
import pandas as pd

from app.extensions import db
from ..models.model import User, UploadFile
from ..forms.form import RegistrationForm, LoginForm, UploadForm
from ..utils.analysis import analyze_file
from ..utils.visualization import generate_charts
from ..utils.ml_model import train_linear_model, predict_new  # ✅ added predict_new

auth = Blueprint("auth", __name__)


@auth.route("/")
def home():
    return render_template("home.html")


@auth.route("/register", methods=["GET", "POST"])
def register():
    form = RegistrationForm()

    if form.validate_on_submit():
        username = form.username.data
        email    = form.email.data
        password = form.password.data

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already registered", "danger")
            return redirect(url_for("auth.register"))

        # ✅ clean single user creation
        user = User(username=username, email=email)
        user.set_password(password)

        db.session.add(user)
        db.session.commit()

        flash("Registration successful. Please login.", "success")
        return redirect(url_for("auth.login"))

    return render_template("register.html", form=form)


@auth.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        email    = form.email.data
        password = form.password.data

        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            session["user_id"]  = user.id
            session["username"] = user.username
            flash("Login successful", "success")
            return redirect(url_for("auth.dashboard"))
        else:
            flash("Invalid email or password", "danger")

    return render_template("login.html", form=form)


@auth.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully", "success")
    return redirect(url_for("auth.login"))


@auth.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("auth.login"))

    form     = UploadForm()
    analysis = None
    charts   = None

    if form.validate_on_submit():
        file     = form.file.data
        filename = str(uuid.uuid4()) + "_" + secure_filename(file.filename)
        upload_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
        file.save(upload_path)

        new_file = UploadFile(
            filename=filename,
            filepath=upload_path,
            user_id=session["user_id"]
        )
        db.session.add(new_file)
        db.session.commit()

        try:
            analysis = analyze_file(upload_path)
            charts   = generate_charts(upload_path)
            flash("File uploaded and analyzed successfully", "success")
        except Exception as e:
            flash(f"Error processing file: {str(e)}", "danger")
            print(f"ERROR: {e}")

    return render_template(
        "dashboard.html",
        form=form,
        analysis=analysis,
        charts=charts
    )


@auth.route("/ml", methods=["GET", "POST"])
def ml_prediction():
    if "user_id" not in session:
        return redirect(url_for("auth.login"))

    user_id   = session["user_id"]
    last_file = UploadFile.query.filter_by(user_id=user_id)\
        .order_by(UploadFile.uploaded_at.desc()).first()

    if not last_file:
        flash("Please upload a file first", "warning")
        return redirect(url_for("auth.dashboard"))

    file_path = last_file.filepath
    df        = pd.read_csv(file_path) if file_path.lower().endswith(".csv") else pd.read_excel(file_path)
    columns   = df.columns.tolist()

    result     = None
    prediction = None

    if request.method == "POST":
        action = request.form.get("action")

        if action == "train":
            feature_cols = request.form.getlist("features")
            target_col   = request.form.get("target")

            if not feature_cols or not target_col:
                flash("Please select at least one feature and a target", "warning")
            elif target_col in feature_cols:
                flash("Target column cannot be a feature too", "warning")
            else:
                try:
                    result = train_linear_model(file_path, feature_cols, target_col)
                    session["ml_result_meta"] = {
                        "feature_cols": feature_cols,
                        "target_col":   target_col,
                        "file_path":    file_path,
                    }
                    flash("Model trained successfully!", "success")
                except Exception as e:
                    flash(f"Training failed: {str(e)}", "danger")

        elif action == "predict":
            meta = session.get("ml_result_meta")
            if not meta:
                flash("Please train the model first", "warning")
            else:
                try:
                    result     = train_linear_model(
                        meta["file_path"], meta["feature_cols"], meta["target_col"]
                    )
                    new_input  = {col: request.form.get(col) for col in meta["feature_cols"]}
                    prediction = predict_new(result, new_input)
                    flash("Prediction complete!", "success")
                except Exception as e:
                    flash(f"Prediction failed: {str(e)}", "danger")

    return render_template(
        "ml.html",
        columns=columns,
        result=result,
        prediction=prediction
    )

