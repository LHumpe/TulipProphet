from flask import Flask
import subprocess

app = Flask(__name__)


@app.route('/')
def hello_world():
    subprocess.call(["pyomo", "solve", "solverApi/src/concrete.py", "--solver=glpk", ])
    return "HELLO"

