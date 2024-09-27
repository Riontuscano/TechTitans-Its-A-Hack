from flask import Flask,render_template,redirect

app=Flask(__name__)

@app.route("/")
def tp():
    return render_template("tp.html")


@app.route("/admin")
def admin_panel():
    return render_template("admin.html")


if __name__=="__main__":
    app.run(debug=True)