from flask import Blueprint, session, request, render_template, redirect, url_for, flash
from ..utils.stock_analyzer import analyze_stock
from ..utils.backtest import run_backtest

stock = Blueprint("stock", __name__)

@stock.route("/stock", methods=["GET", "POST"])
def stock_page():
    if "user_id" not in session:
        return redirect(url_for("auth.login"))

    result = None

    if request.method == "POST":
        symbol = request.form.get("symbol", "").upper().strip()
        try:
            result = analyze_stock(symbol)
        except Exception as e:
            flash(f"Error: {str(e)}", "danger")

    return render_template("stock.html", result=result)



@stock.route("/backtest", methods=["GET", "POST"])
def backtest():
    if "user_id" not in session:
        return redirect(url_for("auth.login"))

    result = None

    if request.method == "POST":
        symbol  = request.form.get("symbol", "").upper().strip()
        capital = request.form.get("capital", "100000")

        try:
            capital = int(capital)
        except ValueError:
            capital = 100000

        try:
            result = run_backtest(symbol, capital)   # ✅ now capital is actually passed
        except Exception as e:
            flash(f"Error: {str(e)}", "danger")

    return render_template("backtest.html", result=result)