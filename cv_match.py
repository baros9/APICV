import docx2txt
import pandas as pd
import os
import numpy as np
import nltk
from pdfminer.high_level import extract_text
import subprocess  # noqa: S404
import sys
import re
#from pyresparser import ResumeParser
from pdfminer.high_level import extract_text
import calendar
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time
# import subprocess  # noqa: S404
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')
from pyresparser import ResumeParser
import spacy
from nltk.corpus import stopwords
nlp = spacy.load('en_core_web_sm')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# cd


PHONE_REG = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
EMAIL_REG = re.compile(r'[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+')
STOPWORDS = set(stopwords.words('english'))
EDUCATION = [
            'BE','B.E.', 'B.E', 'BS', 'B.S',' b.s.', 'b.s.', 'Maîtrise', 'maîtrise',
            'BTECH', 'B.TECH', 'M.TECH', 'MTECH', 'bachlore', "Bachlore's", 'licence','DIC', 'dic',
            'SSC', 'HSC', 'CBSE', 'ICSE', 'X', 'XII', 'master', "master's", 'PhD', 'phd', 'B.S.', 'Masters', 'MASTERS', 'MASTER'
        ]
dates = ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
        '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
mois = ['janvier', 'février', 'mars', 'avril', 'mai', 'juin', 'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre', 
         'january', 'february', 'march', 'april', 'may',  'june', 'july', 'august', 'september', 'october', 'november', 'december',
        'jan.', 'jeb.', 'mar.', 'apr.','Nov.', 'Mai', 'Avr.', 'Sept.','aug.','Aug.','Janv.', 'Oct.'  'sep.', 'oct.', 'nov.', 'dec.', 'janv.', 'févr.', 'avr.', 'juill.', 'sept.', 'oct.', 'nov.', 'déc.', 'present'
      ]
deb = ["experience", "works", "experinces", "expérience", "EXPERIENCE", "EXPÉRIENCE", 'professionnelles']
fin = ["education", "formation", "LEADERSHIP", "leadership", 'RÉALISATIONS', 'certifications', 'langues']
#, 'réalisations'
all_stopwords = stopwords.words('english') + stopwords.words('french')

 
def extract_text_from_pdf(pdf_path):
    text = extract_text(pdf_path)
    text = re.sub(' +', ' ', text)
    text = text.replace('\n\n', ' ').replace('\n', ' ')
    text = text.replace('\xa0', ' ').replace('Nov.2018', 'Nov. 2018').replace('\t', ' ').replace('Aujourd’hui', 'present').replace('aujourd’hui', 'present')
    #text = re.sub(' +', ' ', text)
    text = text.replace('01/', 'janvier ').replace('02/', 'février ').replace('03/', 'mars ').replace('04/', 'avril').replace('05/', 'mai ')
    text = text.replace('06/', 'juin ').replace('07/', 'juillet ').replace('08/', 'août').replace('09/', 'septembre ').replace('10/', 'octobre ').replace('11/', 'november ').replace('12/', 'décembre ') 
    text = text.replace('\uf0d8', '')
    

    return text
def Convetir(text):
  text = text.replace('janvier', 'january').replace('jan.', 'january').replace('janv.', 'january').replace('janv', 'january')
  text = text.replace('févr.', 'february').replace('février', 'february').replace('févr', 'february').replace('févr', 'february')
  text = text.replace('mars.', 'march').replace('mar.', 'march').replace('mars', 'march')
  text = text.replace('Avr.', 'april').replace('avril.', 'april').replace('avr.', 'april')
  text = text.replace('Mai.', 'may').replace('Mai.', 'may').replace('mai.', 'may').replace('mai', 'may')
  text = text.replace('juin.', 'june').replace('juin', 'june')
  text = text.replace('juillet', 'july').replace('Juil.', 'july').replace('juil.', 'july').replace('Juillet', 'july').replace('jul.', 'july')
  text = text.replace('août', 'august').replace('août.', 'august').replace('Août', 'august')
  text = text.replace('septembre', 'september').replace('Sept.', 'september').replace('sept.', 'september')
  text = text.replace('octobre', 'october').replace('oct.', 'october').replace('oct', 'october')
  text = text.replace('novembre', 'november').replace('nov.', 'november').replace('nov', 'november').replace('novemberember', 'november')
  text = text.replace('décembre', 'december').replace('déc.', 'december').replace('dec.', 'december').replace('déc', 'december').replace('dec', 'december')
  text = text.replace('decemberember', 'december')
  # text = text.replace('octobre', 'october').replace('oct.', 'october').replace('oct', 'october')
  # text = text.replace('octobre', 'october').replace('oct.', 'october').replace('oct', 'october')

  return text

def extract_text_from_docx(docx_path):
    txt = docx2txt.process(docx_path)
    txt = re.sub(' +', ' ', txt)

    if txt:
        return txt.replace('\t', ' ').replace('\n\n', ' ').replace('\n\n\n\n', '').replace('\u200b', ' ').replace('\xa0', ' ').replace(' /', '')
    return None


def extract_names(txt):
    person_names = []

    for sent in nltk.sent_tokenize(txt):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                person_names.append(
                    ' '.join(chunk_leave[0] for chunk_leave in chunk.leaves())
                )

    return person_names[0]

def preprocessing(text):
  text = text.split(' ')
  text = [w for w in text if w not in all_stopwords]
  text = ' '.join(text)
  return text

def doc_to_text_catdoc(file_path):
    try:
        process = subprocess.Popen(  # noqa: S607,S603
            ['catdoc', '-w', file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    except (
        FileNotFoundError,
        ValueError,
        subprocess.TimeoutExpired,
        subprocess.SubprocessError,
    ) as err:
        return (None, str(err))
    else:
        stdout, stderr = process.communicate()

    return (stdout.strip(), stderr.strip())


def extract_phone_number(resume_text):
    phone = re.findall(PHONE_REG, resume_text)
    # print(phone)
    if phone:
        number = ''.join(phone[0])
        if resume_text.find(number) >= 0 and len(number) < 16:
          return number#.replace(')', '').replace('(', '')
    return None

def extract_emails(resume_text):
    #print(resume_text)
    return re.findall(EMAIL_REG, resume_text)[0]

# def extract_emails(resume_text):
#     return re.findall(EMAIL_REG, resume_text)


def extract_info(file_folder):
  return ResumeParser(file_folder).get_extracted_data()


def extract_skills(file_folder):
  return ResumeParser(file_folder).get_extracted_data()["skills"]

def extract_degree(file_folder):
  return ResumeParser(file_folder).get_extracted_data()["degree"][0]


def parse_date(x, fmts=("%b %Y", "%B %Y")):
    for fmt in fmts:
        try:
            return datetime.strptime(x, fmt)
        except ValueError:
            pass

def extract_experience(doc, deb, fin):
  windows = []
  doc = doc.lower().split(" ")
  #print(doc)
  for i in range(len(doc)):
    if doc[i] in deb:
      #print(i)
      for j in range(i, len(doc)):
        windows.append(doc[j])
        if doc[j] in fin:
          break
  docx = ' '.join(windows)
  docx = docx.replace('n\n', ' ').replace('\n\n\n\n', ' ')
  return docx

def extract_education(resume_text):
    nlp_text = nlp(resume_text)

    # Sentence Tokenizer
    nlp_text = [sent.string.strip() for sent in nlp_text.sents]

    edu = {}
    # Extract education degree
    for index, text in enumerate(nlp_text):
        for tex in text.split():
            # Replace all special symbols
            tex = re.sub(r'[?|$|.|!|,]', r'', tex)
            if tex.lower() in EDUCATION and tex not in STOPWORDS:
                edu[tex] = text + nlp_text[index + 1]

    # Extract year
    education = []
    for key in edu.keys():
        year = re.search(re.compile(r'(((20|19)(\d{2})))'), edu[key])
        if year:
            education.append((key, ''.join(year[0])))
        else:
            education.append(key)
    if len(education) == 0 :
      return None
    elif type(education[0]) == tuple :
      return education[0][0]
    else:
      return education[0]

def format_date(text):
  text = text.replace('current', 'present')

  text = text.split(' ')
  text = [word for word in text if not word in all_stopwords]

  text = list(filter(None, text))
  for i in range(len(text) - 1):
    text[i] = text[i].replace('/', '–')
    if (text[i] in dates and text[i+ 1] in mois) :
      text.insert(i+1, '–')
  return text

def year_experience(cv):
  months = "|".join(calendar.month_abbr[1:] + calendar.month_name[1:])
  pattern = fr"(?i)((?:{months}) *\d{{4}}) *(?:-|–) *(present|(?:{months}) *\d{{4}})"
  total_experience = None

  for start, end in re.findall(pattern, cv):
      if end.lower() == "present" or end.lower() == 'Aujourd’hui' or end.lower() == 'Depuis'.lower():
          today = datetime.today()
          end = f"{calendar.month_abbr[today.month]} {today.year}"

      duration = relativedelta(parse_date(end), parse_date(start))

      if total_experience:
          total_experience += duration
      else: 
          total_experience = duration

      #print(f"{start}-{end} ({duration.years} years, {duration.months} months)")
  #print(total_experience.years)
  if total_experience:
      #print(f"total experience:  {total_experience.years} years, {total_experience.months} months")
      years = str(total_experience.years) + " " + " ans" + "," + " " +  str(total_experience.months)  + " " + " mois"
  else:
      years = None
  return years

def nom_cv(path):
  Nom_cv = []
  os.chdir(path)
  rep = os.walk(path)
  for ath, dirs, files in rep:
    Nom_cv.append(files)
  return Nom_cv[0]

def load_rep(path):
  os.chdir(path)
  rep = os.walk(path)
  w = []
  for ath, dirs, files in rep:
    print(' ')
    for i in range(len(files)):
      if (os.path.splitext(files[i])[1].lower()=='.docx'):
        w.append(extract_text_from_docx(files[i]))
      if (os.path.splitext(files[i])[1].lower()=='.pdf'):
        w.append(extract_text_from_pdf(files[i]))
  return w
# def format_date_cor(L):
#   L = L.split(' ')
#   for i in range(len(L)):

# def folder_table(folder, deb, fin):
#   Ncv = nom_cv(folder)
#   text = load_rep(folder)
#   w = [[] for i in range(len(Ncv))]
#   for i in range(len(text)):
#     text_cut = extract_experience(text[i],  deb, fin)
#     w[i].append(nom_cv(folder)[i])
#     w[i].append(year_experience(text_cut))
#     #w[].append(extract_names(text))
#     w[i].append(extract_phone_number(text[i]))
#    # w[i].append(extract_emails(text[i]))
#     #w[i].append(extract_education(text)[i])
#     #w.append(files)
#     # skills = extract_skills(folder_path)
    
#   #L = [Nom_cv, experience, degree, phone, email]
#   L = w
#   df = pd.DataFrame(L)
#   df.columns =['Nom_cv', 'Years of Experience', 'Phone']
#   return df

    
# def data_table(folder_path,folder, deb, fin):
#   #text = extract_text_from_docx(folder_path)
#   text = extract_text_from_pdf(folder_path)
#   name = extract_names(text)
#   #email = extract_emails(text)
#   phone = extract_phone_number(text)
#   #degree = extract_education(text)
#  # print(degree)
#   Nom_cv = nom_cv(folder)
#   skills = extract_skills(folder_path)
#   text_cut = extract_experience(text,  deb, fin)
#   experience = year_experience(text_cut)
#   L = [Nom_cv, experience, phone]
#   df = pd.DataFrame([L])
#   df.columns =['Nom_cv','Years of Experience', 'Phone']
#   return df




def Recommandation_CV(folder,path_offre, deb, fin):
  jobs = docx2txt.process(path_offre)
  Ncv = nom_cv(folder)
  #print(Ncv[0])
  text = load_rep(folder)
  w = [[] for i in range(len(Ncv))]
  for i in range(len(text)):
    path_file = folder + Ncv[i]
    text_cut = extract_experience(text[i],  deb, fin)
    text_cut = Convetir(text_cut)
    text_cut = format_date(text_cut)
    text_cut = ' '.join(text_cut)
    w[i].append(nom_cv(folder)[i])
    w[i].append(year_experience(text_cut))
    #w[].append(extract_names(text))
    w[i].append(extract_education(text[i]))
    #w[i].append(Score(path_file, path_offre))
    w[i].append(Score(text[i], jobs))

    w[i].append(extract_phone_number(text[i]))
    w[i].append(extract_emails(text[i]))

    #w.append(files)
    # skills = extract_skills(folder_path)
  #print(w)
  #L = [Nom_cv, experience, degree, phone, email]
  L = w

  df = pd.DataFrame(L)
  df.columns =['Nom', "Annee d'éxperience", "Niveau d'etude",'Score(en %)', 'Telephone', 'Email']

  #df  = df.sort_values(by=["Score(en %)"], ascending = False)
  return df.to_dict()

def transform(text):
  text.replace('01/', "january ").replace('1/', 'january ')
  text.replace('02/', "february ").replace('2/', 'february ')
  text.replace('03/', "march ").replace('3/', 'march ')
  text.replace('04/', "april ").replace('4/', 'april ')
  text.replace('05/', "may ").replace('5/', 'may ')
  text.replace('06/', "june ").replace('5/', 'june ')
  text.replace('07/', "july ").replace('7/', 'july ')
  text.replace('08/', "august ").replace('9/', 'september ')
  text.replace('09/', "september ").replace('1/', 'january ')
  text.replace('10/', "october ")
  text.replace('11/', "november ")
  text.replace('12/', "december ")

  return text
    
def data_table(folder_path,folder, deb, fin):
  #text = extract_text_from_docx(folder_path)
  text = extract_text_from_pdf(folder_path)
  # print(text)
  text = text.replace('november2018', 'november 2018')
  #print(text)
  #name = nom_cv(folder_path)
  email = extract_emails(text)
  phone = extract_phone_number(text)
  #degree = extract_education(text)
 # print(degree)
  Nom_cv = nom_cv(folder)[1]
  skills = extract_skills(folder_path)
  text_cut = extract_experience(text,  deb, fin)
  #text_cut = text_cut.replace('november2018', 'november 2018')
  text_cut = format_date(text_cut)
  text_cut = ' '.join(text_cut)
  
  text_cut = Convetir(text_cut)
  #print(text_cut)
  experience = year_experience(text_cut)
  #print(experience)
  L = [Nom, experience, phone, email]
  df = pd.DataFrame([L])
  #df  = df.sort_values(by=["Score(en %)"], ascending = False)
  return df.to_dict()

# def Score(cvs_path, jobs_path):
#   cvs = extract_skills(cvs_path)
#   cvs = ' '.join(cvs)
#   jobs = extract_skills(jobs_path)
#   jobs = ' '.join(jobs)
#   text = [cvs, jobs]
#   cv = CountVectorizer()
#   count_matrix = cv.fit_transform(text)
#   t = cosine_similarity(count_matrix)
#   matchPercentage = cosine_similarity(count_matrix)[0][1] * 100
#   matchPercentage = round(matchPercentage, 2) # round to two decimal
#   return matchPercentage

def Score(cvs, jobs):
  text = [cvs, jobs]
  cv = CountVectorizer()
  count_matrix = cv.fit_transform(text)
  t = cosine_similarity(count_matrix)
  matchPercentage = cosine_similarity(count_matrix)[0][1] * 100
  matchPercentage = round(matchPercentage, 2) # round to two decimal
  return matchPercentage



