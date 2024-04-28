from flask import Flask,render_template,request,jsonify
import bot2 as bot2

app = Flask(__name__)

@app.route("/",methods=['GET'])
def root():
    return render_template('index.html')

@app.route("/bot",methods=['POST'])
def calc():
    data_obtained = request.json
    result = bot2.get_result(data_obtained)
    return jsonify(result)

if (__name__) == "__main__":
    app.run( debug=True, host="0.0.0.0") #widelisting