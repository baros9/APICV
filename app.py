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
    return " API On...."
@app.route('/predict',methods=['GET','POST'])
def RecommandationCV():
    # data =  json.loads(dataCV)
    path_cv = request.args["path_cv"]
    #print(path_cv)
    path_offre = request.args["path_offre"]
    # path_cv = data['path_cv']
    # path_offre = data['path_offre']

    deb = ["experience", "works", "experinces", "expérience", "EXPERIENCE", "EXPÉRIENCE", 'professionnelles']
    fin = ["education", "formation", "LEADERSHIP", "leadership", 'RÉALISATIONS', 'certifications', 'langues']
    cv_recommandation = matchs.Recommandation_CV(path_cv, path_offre, deb, fin)
    cv_recommandation = dict(cv_recommandation)
    cv_recommandation = pd.DataFrame(cv_recommandation)
    print(cv_recommandation['Nom'])
    cv_recommandation  = cv_recommandation.sort_values(by=["Score(en %)"], ascending = False)
    return  render_template('tested.html', column_names=cv_recommandation.columns.values, row_data=list(cv_recommandation.values.tolist()), zip=zip)
    #, prediction_text='Le modele predit  :{}'.format(cv_recommandation.to_html())
   # cv_recommandation =cv_recommandation.to_ # use pandas method to auto generate html

    #return cv_recommandation.to
    
    #return render_template('/home/abdoulaye/Desktop/ML_APP/index.html', prediction_text='Le modele predit  :{}'.format(output))



if __name__ == "__main__":
    app.run(debug=True)
