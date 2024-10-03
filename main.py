import pandas as pd
import pickle
from utils import clean_text
from utils import vectorize
from transformers import AutoModelForSequenceClassification


def check_text(input, model):
    text_news = vectorize(pd.Series([clean_text(str(input))], name='text'))
    pred = model.predict_proba(text_news)
    return pred[0][0]


def run_model(input: str, option: str):
    if option == "Logistic Regression":
        model = pickle.load(open('./models/logreg.pkl', 'rb'))
    elif option == "Random Forest Classifier":
        model = pickle.load(open('./models/random_forest.pkl', 'rb'))
    elif option == "Support Vector Classifier":
        model = pickle.load(open('./models/svc.pkl', 'rb'))
    elif option == "CatBoost":
        model = pickle.load(open('./models/catboost.pkl', 'rb'))
    else:
        print('Ошибка загрузки модели')

    if model:
        try:
            ans = check_text(input, model)
            strin = "Новость достоверна с вероятностью " + str(round(ans * 100)) + "%."
        except:
            strin = "Что-то не так с вашей новостью"

        print(strin)


if __name__ == '__main__':
    input = 'В Архангельской области прошел Форум молодых ученых "Полюс"'
    # Ввод текста новости

    option = 'CatBoost'
    # Можно менять модель из следующих доступных:
    # ('Logistic Regression', 'Random Forest Classifier',
    # 'Support Vector Classifier', 'CatBoost')

    run_model(input, option)
