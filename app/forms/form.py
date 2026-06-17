from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length, Email, EqualTo


class RegistrationForm(FlaskForm):
    username = StringField(
        "username",
        validators=[
            DataRequired(),
            Length(min=3, max=25)
        ]
    )

    email = StringField(
        "email",
        validators=[
            DataRequired(),
            Email()
        ]
    )

    password = PasswordField(
        "password",
        validators=[
            DataRequired(),
            Length(min=8)
        ]
    )

    confirm_password = PasswordField(
        "confirm_password",
        validators=[
            DataRequired(),
            EqualTo("password", message="Passwords must match")
        ]
    )

    submit = SubmitField("Register")


class LoginForm(FlaskForm):
    email = StringField(
        "email",
        validators=[DataRequired(), Email()]
    )

    password = PasswordField(
        "password",
        validators=[DataRequired()]
    )

    submit = SubmitField("Login")


class UploadForm(FlaskForm):
    file = FileField(
        "Upload Excel/CSV File",
        validators=[
            FileRequired(),
            FileAllowed(["csv", "xlsx"], "Only CSV or Excel files allowed")
        ]
    )

    submit = SubmitField("Upload")