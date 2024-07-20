import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
import csv
import warnings
from deep_translator import GoogleTranslator
from langdetect import detect

warnings.filterwarnings("ignore", category=DeprecationWarning)

translator = GoogleTranslator()

def readn(nstr, lang='en'):
    engine = pyttsx3.init()
    if lang == 'en':
        engine.setProperty('voice', 'english+f5')
    elif lang == 'es':
        engine.setProperty('voice', 'spanish')
    engine.setProperty('rate', 130)
    engine.say(nstr)
    engine.runAndWait()
    engine.stop()

def translate_text(text, src='en', dest='en'):
    return translator.translate(text, source=src, target=dest)

def detect_language(text):
    return detect(text)

training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

reduced_data = training.groupby(training['prognosis']).max()

le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)

clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)
scores = cross_val_score(clf, x_test, y_test, cv=3)
print(scores.mean())

model = SVC()
model.fit(x_train, y_train)
print("for svm: ")
print(model.score(x_test, y_test))

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

severityDictionary = {}
description_list = {}
precautionDictionary = {}
symptoms_dict = {}

for index, symptom in enumerate(x):
    symptoms_dict[symptom] = index

def calc_condition(exp, days, lang='en'):
    sum_severity = sum(severityDictionary[item] for item in exp)
    condition = "You should take the consultation from doctor." if (sum_severity * days) / (len(exp) + 1) > 13 else "It might not be that bad but you should take precautions."
    condition = translate_text(condition, dest=lang)
    print(condition)
    readn(condition, lang)

def getDescription():
    global description_list
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            description_list[row[0]] = row[1]

def getSeverityDict():
    global severityDictionary
    with open('MasterData/symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) < 2:
                continue  # Skip rows that don't have enough columns
            try:
                severityDictionary[row[0]] = int(row[1])
            except ValueError:
                print(f"Invalid data: {row}")
                continue  # Skip rows with non-integer severity values

def getprecautionDict():
    global precautionDictionary
    with open('MasterData/symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            precautionDictionary[row[0]] = row[1:5]

def getInfo():
    print("-----------------------------------HealthCare ChatBot-----------------------------------")
    print("\nYour Name? \t\t\t\t", end="->")
    name = input("")
    print("Hello, ", name)

def check_pattern(dis_list, inp):
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    return (1, pred_list) if pred_list else (0, [])

def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[symptoms_dict[item]] = 1
    return rf_clf.predict([input_vector])

def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))

def tree_to_code(tree, feature_names, lang='en'):
    tree_ = tree.tree_
    feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]
    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []

    while True:
        print("\nEnter the symptom you are experiencing  \t\t", end="->")
        disease_input = input("")
        lang = detect_language(disease_input)
        disease_input_translated = translate_text(disease_input, src=lang, dest='en')
        conf, cnf_dis = check_pattern(chk_dis, disease_input_translated)
        if conf == 1:
            print("searches related to input: ")
            for num, it in enumerate(cnf_dis):
                print(num, ")", it)
            conf_inp = int(input(f"Select the one you meant (0 - {num}):  ")) if num != 0 else 0
            disease_input = cnf_dis[conf_inp]
            break
        else:
            print("Enter valid symptom.")

    num_days = int(input("Okay. From how many days ? : "))

    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            if name == disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            print("Are you experiencing any ")
            symptoms_exp = [syms for syms in list(symptoms_given) if input(f"{syms}? : ") == "yes"]
            second_prediction = sec_predict(symptoms_exp)
            calc_condition(symptoms_exp, num_days, lang)
            if present_disease[0] == second_prediction[0]:
                disease_description = description_list[present_disease[0]]
                disease_description_translated = translate_text(disease_description, dest=lang)
                print("You may have ", present_disease[0])
                print(disease_description_translated)
                readn(f"You may have {present_disease[0]}", lang)
                readn(disease_description_translated, lang)
            else:
                disease_description_1 = description_list[present_disease[0]]
                disease_description_2 = description_list[second_prediction[0]]
                disease_description_translated_1 = translate_text(disease_description_1, dest=lang)
                disease_description_translated_2 = translate_text(disease_description_2, dest=lang)
                print("You may have ", present_disease[0], "or ", second_prediction[0])
                print(disease_description_translated_1)
                print(disease_description_translated_2)
                readn(f"You may have {present_disease[0]} or {second_prediction[0]}", lang)
                readn(disease_description_translated_1, lang)
                readn(disease_description_translated_2, lang)
            precaution_list = precautionDictionary[present_disease[0]]
            precaution_list_translated = [translate_text(prec, dest=lang) for prec in precaution_list]
            print("Take following measures : ")
            for i, j in enumerate(precaution_list_translated):
                print(i + 1, ")", j)
                readn(j, lang)

    recurse(0, 1)

getSeverityDict()
getDescription()
getprecautionDict()
getInfo()
tree_to_code(clf, cols)
