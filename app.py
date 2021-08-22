from os import path, pipe
import flask
from flask import Flask, jsonify,make_response, current_app, request, url_for, render_template
app = flask.Flask(__name__)
app.config["DEBUG"] = True
from flask_cors import CORS
from functools import update_wrapper, wraps
import cv_match as matchs
import json

import pandas as pd
#!pip3 install pyresparser

#!!python -m spacy download fr_core_news_sm

#!python -m spacy download fr_core_news_md
app = Flask(__name__)
CORS(app)
cors = CORS(app, resources  = {
    r"/*" : {
        "origins" : "*"
    }
}

)

# main index page route
# @app.route('/')
# def home():
#     return render_template('/home/abdoulaye/Desktop/ML_APP/index.html')
@app.route('/')
def home():
    return 'API start ....'
@app.route('/predict',methods=['GET'])
def RecommandationCV():
    # data =  json.loads(dataCV)
    path_cv = request.args["path_cv"]
    print(path_cv)
    path_offre = request.args["path_offre"]
    # path_cv = data['path_cv']
    # path_offre = data['path_offre']

    deb = ["experience", "works", "experinces", "expérience", "EXPERIENCE", "EXPÉRIENCE", 'professionnelles']
    fin = ["education", "formation", "LEADERSHIP", "leadership", 'RÉALISATIONS', 'certifications', 'langues']
    cv_recommandation = matchs.Recommandation_CV(path_cv, path_offre, deb, fin)
    return render_template('table.html')
    
    
    # #return render_template('/home/abdoulaye/Desktop/ML_APP/index.html', prediction_text='Le modele predit  :{}'.format(output))



if __name__ == "__main__":
    app.run(debug=True)
