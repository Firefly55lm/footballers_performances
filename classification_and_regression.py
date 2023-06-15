import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, model_selection
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, r2_score, mean_squared_error
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

def float_encoding(string):
    '''
        Convertion function
    '''
    if string == "positive":
        return 2
    elif string == "neutral":
        return 1
    else:
        return 0

def calc_delta(a, b):
    '''
        Calculates Delta between two numbers a and b keeping the sign of the variation
    '''
    if a > b:
        delta = b - a
    else:
        delta = a - b
        if delta < 0:
            delta = - delta
    return delta

def level_polarity(num):
    '''
        Identifies the polarity for given number (positive, negative, null)
    '''
    if float(num) == 0:
        return "No Variation"
    elif float(num) > 0:
        return "Positive Variation"
    else:
        return "Negative Variation"


def model_name(text):
    '''
        Extracts the name of the model from a standard text
    '''
    try:
        return str(text).split()[4].replace("(", "").replace(")", "").replace("]", "").replace(",", "")
    except:
        return "(Name Not Available)"

def model_name_reg(text):
    '''
        Extracts the name of the model from a standard text. Customized for regression model
    '''
    try:
        return str(text).split()[7].replace("(", "").replace(")", "").replace("]", "").replace(",", "")
    except:
        return "(Name Not Available)"

def extract_model_informations(fav_mod, y, data):
    '''
        Extracts the informations of the given model
        :param fav_mod: model
        :param y: target variable
        :return: prints informations
    '''
    print("\n\033[36mRISULTATI\033[0m")
    name_best = model_name(fav_mod)

    if name_best == "LinearDiscriminantAnalysis":
        best_mod = fav_mod.best_estimator_
        selected_features = data.columns[best_mod.named_steps['selector'].get_support()]
        print(
            f"\n\033[01mL'accuracy migliore con y = {y} è del modello {name_best} con un punteggio di {best_acc} di accuracy\033[0m\n",
            f"\033[01m\nVariabili utilizzate:\033[0m\n {tuple(selected_features)}\033[0m\n",
            f"\033[01m\nDettagli modello:\033[0m\n",
            best_mod, '\n')

    elif name_best == "SVC":
        best_mod = fav_mod.best_estimator_
        selected_features = data.columns[best_mod.named_steps['selector'].get_support(indices=True)].tolist()
        print(
            f"\n\033[01mL'accuracy migliore con y = {y} è del modello {name_best} con un punteggio di {best_acc} di accuracy\033[0m\n",
            f"\033[01m\nVariabili utilizzate:\033[0m\n {selected_features}\033[0m\n",
            f"\033[01m\nDettagli modello:\033[0m\n",
            best_mod, '\n')
    else:
        best_mod = fav_mod.best_estimator_
        print(
            f"\n\033[01mL'accuracy migliore con y = {y} è del modello {name_best} con un punteggio di {best_acc} di accuracy\033[0m\n",
            f"\033[31m\nNon sono disponibili informazioni sulle variabili selezionate\033[0m\n",
            f"\033[01m\nDettagli modello:\033[0m\n",
            best_mod, '\n')

def extract_model_informations_reg(fav_mod,y):
    '''
        Extracts the informations of the given regression model
        :param fav_mod: model
        :param y: target variable
        :return: prints informations
    '''
    print("\n\033[36mRISULTATI\033[0m")
    name_best = model_name_reg(fav_mod)

    if name_best == "LinearRegression":
        best_mod = fav_mod.best_estimator_
        selected_features = x_data_reg.columns[best_mod.named_steps['selector'].get_support()]
        print(
            f"\n\033[01mL'MSE migliore con y = {y} è del modello {name_best} con un punteggio di {best_score} di MSE"
            f" e un R2 di {best_R2}\033[0m\n",
            f"\033[01m\nVariabili utilizzate:\033[0m\n {tuple(selected_features)}\033[0m\n",
            f"\033[01m\nDettagli modello:\033[0m\n",
            best_mod, '\n')

    elif name_best == "xgb.XGBRegressor":
        best_mod = fav_mod.best_estimator_
        selected_features = x_data_reg.columns[best_mod.named_steps['selector'].get_support(indices=True)].tolist()
        print(
            f"\n\033[01mL'MSE migliore con y = {y} è del modello {name_best} con un punteggio di {float(best_score)} di MSE"
            f"e un R2 di {best_R2}\033[0m\n",
            f"\033[01m\nVariabili utilizzate:\033[0m\n {selected_features}\033[0m\n",
            f"\033[01m\nDettagli modello:\033[0m\n",
            best_mod, '\n')
    else:
        best_mod = fav_mod.best_estimator_
        print(
            f"\n\033[01mL'MSE migliore con y = {y} è del modello {name_best} con un punteggio di {float(best_score)} di MSE"
            f"e un R2 di {best_R2}\033[0m\n",
            f"\033[31m\nNon sono disponibili informazioni sulle variabili selezionate\033[0m\n",
            f"\033[01m\nDettagli modello:\033[0m\n",
            best_mod, '\n')


if __name__ == '__main__':
    df = pd.read_csv("premier_with_sentiment.csv")
    print(df.keys())
    df = df.dropna()
    print(len(df))


    # PREPARING DATA
    df["float_emotion_vader"] = df["vader_emotion_before"].apply(float_encoding)
    df["float_emotion_tb"] = df["tb_emotion_before"].apply(float_encoding)


    x_data = df[["tb_polarity_before", "float_emotion_tb",
                "float_emotion_vader", "vader_polarity_before", "goal", "assist",
                "minutes"]]
    y_data = df[["char_rating"]]
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x_data, y_data, test_size=0.33, random_state=42)


    # MODELS FOR Y = RATING CHAR (SENTIMENT BEFORE TO GAME RATING)

    # SVC Pipeline
    pipeline_svc = Pipeline(
        [
            ('selector', SelectKBest(f_classif)),
            ('model', svm.SVC())
        ]
    )
    parameters_svc = {
        'selector__k': [1, 2, 3, 4],
        'model__kernel': ['linear', 'rbf'],
        'model__C': [1, 10],
        'model__probability': [True]}

    svc = model_selection.GridSearchCV(
        estimator=pipeline_svc,
        param_grid=parameters_svc,
        n_jobs=-1,
        scoring="f1_micro",
        cv=5,
        verbose=2
    )

    # LDA Pipeline
    pipeline_lda = Pipeline(
        [
         ('selector',SelectKBest(f_classif)),
         ('model',LinearDiscriminantAnalysis())
        ]
    )
    parameters_lda = {
            'selector__k':[1,2,3,4,5],
            "model__solver":['svd', 'lsqr', 'eigen']}

    lda = model_selection.GridSearchCV(
        estimator = pipeline_lda,
        param_grid = parameters_lda,
        n_jobs=-1,
        scoring="f1_micro",
        cv=5,
        verbose=2
    )

    # Running models
    models = (svc, lda)
    parameters = (parameters_svc, parameters_lda)

    acc = []
    best_acc = 0
    c =0
    for p in models:
        print("Performing grid search...")
        p.fit(x_train, np.ravel(y_train, order="C"))
        print("Best score: %0.3f" % p.best_score_)
        print("Best parameters set:")
        best_parameters = p.best_estimator_.get_params()
        if p == svc:
            for param_name in sorted(parameters_svc.keys()):
                print("\t%s: %r" % (param_name, best_parameters[param_name]))
        else:
            for param_name in sorted(parameters_lda.keys()):
                print("\t%s: %r" % (param_name, best_parameters[param_name]))
        predicted = p.predict(x_test)
        acc_mod = accuracy_score(y_test, predicted)
        acc.append((p,acc_mod))
        print("Accuracy:",accuracy_score(y_test, predicted))
        print(precision_recall_fscore_support(y_test, predicted))
        print(classification_report(y_test, predicted))
        conf = confusion_matrix(y_test, predicted, labels=p.classes_)
        disp2 = ConfusionMatrixDisplay(conf, display_labels=p.classes_)
        disp2.plot()
        if c == 0:
            plt.title("Confusion Matrix della SVC y = char_rating")
            c+=1
        else:
            plt.title("Confusion Matrix della LDA con y = char_rating")
        plt.tight_layout()
        plt.show()
    for el in acc:
        if el[1] > best_acc:
            best_acc = el[1]
            fav_mod = el[0]
        else:
            continue

    # Printing the results
    extract_model_informations(fav_mod, "char_rating", x_data)

    # MODELS FOR Y = VADER POLARITY CHAR (GAME TO SENTIMENT AFTER)

    df["delta_pol_vad"] = df.apply(lambda df: calc_delta(df['vader_polarity_before'], df['vader_polarity_after']),
                            axis=1)

    df["char_delta_pol_vad"] = df["delta_pol_vad"].apply(level_polarity)

    x_data2 = df[["vader_polarity_before","rating", "goal", "minutes", "assist"]]
    y_data2 = df[["char_delta_pol_vad"]]
    x_train2, x_test2, y_train2, y_test2 = model_selection.train_test_split(
    x_data2, y_data2, test_size=0.33, random_state=45)

    # Running models
    acc = []
    best_acc = 0
    c = 0
    for p in models:
        print("Performing grid search...")
        p.fit(x_train2, np.ravel(y_train2, order="C"))
        print("Best score: %0.3f" % p.best_score_)
        print("Best parameters set:")
        best_parameters = p.best_estimator_.get_params()
        if p == svc:
            for param_name in sorted(parameters_svc.keys()):
                print("\t%s: %r" % (param_name, best_parameters[param_name]))
        else:
            for param_name in sorted(parameters_lda.keys()):
                print("\t%s: %r" % (param_name, best_parameters[param_name]))
        predicted = p.predict(x_test2)
        acc_mod = accuracy_score(y_test2, predicted)
        acc.append((p,acc_mod))
        print("Accuracy:",accuracy_score(y_test2, predicted))
        print(precision_recall_fscore_support(y_test2, predicted))
        print(classification_report(y_test2, predicted))
        conf = confusion_matrix(y_test2, predicted, labels=p.classes_)
        disp2 = ConfusionMatrixDisplay(conf, display_labels=p.classes_)
        disp2.plot()
        if c == 0:
            plt.title("Confusion Matrix della SVC y = char_delta_pol_vad")
            c+=1
        else:
            plt.title("Confusion Matrix della LDA con y = char_delta_pol_vad")
        plt.tight_layout()
        plt.show()
    for el in acc:
        if el[1] > best_acc:
            best_acc = el[1]
            fav_mod = el[0]
        else:
            continue

    # Printing results
    extract_model_informations(fav_mod, "char_delta_pol_vad", x_data2)


    # REGRESSION MODELS

    # Defining pipelines
    pipeline_linreg = Pipeline(
        [
         ('selector',SelectKBest(f_regression)),
         ('model',LinearRegression())
        ]
    )

    pipeline_XGBoost_reg = Pipeline(
        [
         ('selector',SelectKBest(f_regression)),
         ('model',xgb.XGBRegressor())
        ]
    )

    parameters_lin = {
            'selector__k':[1,2,3,4],
            "model__positive":[True, False]
    }

    parameters_xgreg = {
            'selector__k':[1,2,3,4]
    }

    print("\nregression")

    lin = model_selection.GridSearchCV(
        pipeline_linreg,
        parameters_lin,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        cv=5,
        verbose=2
    )

    xgreg = model_selection.GridSearchCV(
        pipeline_XGBoost_reg,
        parameters_xgreg,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        cv=5,
        verbose=2
    )

    models_reg = [lin, xgreg]
    parameters = (parameters_svc, parameters_lda)


    y_data_reg = df[["delta_pol_vad"]]
    x_data_reg = df[["vader_polarity_before","rating", "goal",
                     "assist", "minutes"]]

    x_train_reg, x_test_reg, y_train_reg, y_test_reg = model_selection.train_test_split(
    x_data_reg, y_data_reg, test_size=0.33, random_state=45)


    # Running models
    score = []
    best_score = 1000
    for p in models_reg:
        print("Performing grid search...")
        p.fit(x_train_reg,np.ravel(y_train_reg, order= "C"))
        print("Best score: %0.3f" % p.best_score_)
        print("Best parameters set:")
        best_parameters = p.best_estimator_.get_params()
        if p == lin:
            for param_name in sorted(parameters_lin.keys()):
                print("\t%s: %r" % (param_name, best_parameters[param_name]))
        else:
            for param_name in sorted(parameters_xgreg.keys()):
                print("\t%s: %r" % (param_name, best_parameters[param_name]))
        predicted = p.predict(x_test_reg)
        MSE = mean_squared_error(y_test_reg, predicted)
        R2= r2_score(y_test_reg, predicted)
        score.append((p,MSE, R2))
    for el in score:
        if el[1] < best_score:
            best_score = el[1]
            best_R2 = el[2]
            fav_mod = el[0]
        else:
            continue

    # Printing results
    extract_model_informations_reg(fav_mod, "delta_pol_vad")

    # END




