import pandas as pd
import pyLDAvis
import pyLDAvis.sklearn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

def clean_words_remaining(text):
    """
        Deletes useless words from a given text
    """
    words = ['a href', 'href', 'search q', 'HARRY', 'MAGUIRE', 'ALEXANDER', 'ISAAK', 'ISAK', 'ERLING', 'BRAUT', 'HAALAND', 'DARWIN', 'NUNEZ',
             'MYKHAYLO', 'MUDRYK', 'KAORU', 'MITOMA', 'BUKAYO', 'SAKA', 'JAMES', 'MADDISON', 'HARRY', 'KANE', 'MARCUS', 'RASHFORD',
             'ARSENAL', 'MANCHESTER', 'CITY', 'UTD', 'UNITED', 'NEWCASTLE', 'LEICESTER', 'BRIGHTON', 'LIVERPOOL', 'CHELSEA', 'REDS',
             'TOTTENHAM', 'HOTSPURS', 'SPURS', 'NUFC', 'MOHAMED', 'SALAH', 'BRENDAN', 'RODGERS', 'HOWE', 'EDDIE', 'EDDY', 'URUGUAY',
             'ENGLAND', 'BRAZIL', 'MARTINELLI', 'CASEMIRO', 'RONALDO', 'FOXES', 'CALLUM', 'WILSON', 'BRUNO']
    for w in words:
        text = text.lower().replace(w.lower(), "")
    return text

def display_topics(model, feature_names, no_top_words):
    """
        Displays found topics
    """
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
            for i in topic.argsort()[:-no_top_words - 1:-1]]))

def clean_set(set):
    """
        Applies function clean_words_remaining to a dataframe column
    """
    df_temp = pd.DataFrame(set)
    df_temp = df_temp['text'].apply(clean_words_remaining)
    return df_temp

def topic_modelling(datasets):
    """
        The main function: carries out the topic modelling analysis
    """
    for item in datasets:
        dataset = clean_set(item[1])
        name = item[0]
        print(f"\n\033[36mTOPIC MODELLING FOR {name} SENTIMENT\033[0m\n")
        no_features = 1000
        tf_vectorizer = CountVectorizer(max_df=0.95,
                                            min_df=2,
                                            max_features=no_features,
                                            stop_words='english')
        tf = tf_vectorizer.fit_transform(dataset)
        tf_feature_names = tf_vectorizer.get_feature_names_out()
        no_topics = 10


        lda_model = LatentDirichletAllocation(
                n_components=no_topics,
                max_iter=5,
                learning_method='online',
                learning_offset=50.,
                random_state=0
            ).fit(tf)

        print("Topic Modeling con LDA")
        display_topics(lda_model, tf_feature_names, 20)


        lda = LatentDirichletAllocation(random_state=0)
        searchParams = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}       #n components 10,15,20,25,30
        model = GridSearchCV(lda,
                                 param_grid=searchParams,
                                 verbose=3,
                                 n_jobs=-1)
        model.fit(tf)
        best_lda_model = model.best_estimator_
        print(model)
        print("Best Log Likelihood Score: ",model.best_score_)
        print("Best Model's Params: ", model.best_params_)
        print("Model Perplexity: ", best_lda_model.perplexity(tf))


        n_topics = [10, 15, 20, 25, 30]
        log_likelyhoods_5 = [round(model.cv_results_['mean_test_score'][index])
                             for index, gscore in enumerate(model.cv_results_['params']) if gscore['learning_decay']==0.5]
        log_likelyhoods_7 = [round(model.cv_results_['mean_test_score'][index])
                             for index, gscore in enumerate(model.cv_results_['params']) if gscore['learning_decay']==0.7]
        log_likelyhoods_9 = [round(model.cv_results_['mean_test_score'][index])
                             for index, gscore in enumerate(model.cv_results_['params']) if gscore['learning_decay']==0.9]

        # Show graph
        plt.figure(figsize=(12, 8))
        plt.plot(n_topics, log_likelyhoods_5, label='0.5')
        plt.plot(n_topics, log_likelyhoods_7, label='0.7')
        plt.plot(n_topics, log_likelyhoods_9, label='0.9')
        plt.title("Choosing Optimal LDA Model")
        plt.xlabel("Num Topics")
        plt.ylabel("Log Likelyhood Scores")
        plt.legend(title='Learning decay', loc='best')
        plt.show()


        panel = pyLDAvis.sklearn.prepare(lda_model=best_lda_model, dtm=tf, vectorizer=tf_vectorizer)
        # panel

        name = name.lower()
        pyLDAvis.save_html(panel, f'LDA_panel_{name}.html')


if __name__ == '__main__':
    dataset = pd.read_csv("sentiment-dataset.csv", encoding="windows-1252")

    dataset_neg = dataset["text"][
        (dataset["vader_emotion"] == "negative") & ((dataset["player_name"] == "HARRY MAGUIRE") |
                                                    (dataset["player_name"] == "ALEXANDER ISAK") |
                                                    (dataset["player_name"] == "ERLING BRAUT HAALAND") |
                                                    (dataset["player_name"] == "DARWIN NUNEZ") |
                                                    (dataset["player_name"] == "MYKHAYLO MUDRYK") |
                                                    (dataset["player_name"] == "KAORU MITOMA") |
                                                    (dataset["player_name"] == "BUKAYO SAKA") |
                                                    (dataset["player_name"] == "JAMES MADDISON") |
                                                    (dataset["player_name"] == "HARRY KANE") |
                                                    (dataset["player_name"] == "MARCUS RASHFORD")
                                                    )]
    dataset_pos = dataset["text"][
        (dataset["vader_emotion"] == "positive") & ((dataset["player_name"] == "HARRY MAGUIRE") |
                                                    (dataset["player_name"] == "ALEXANDER ISAK") |
                                                    (dataset["player_name"] == "ERLING BRAUT HAALAND") |
                                                    (dataset["player_name"] == "DARWIN NUNEZ") |
                                                    (dataset["player_name"] == "MYKHAYLO MUDRYK") |
                                                    (dataset["player_name"] == "KAORU MITOMA") |
                                                    (dataset["player_name"] == "BUKAYO SAKA") |
                                                    (dataset["player_name"] == "JAMES MADDISON") |
                                                    (dataset["player_name"] == "HARRY KANE") |
                                                    (dataset["player_name"] == "MARCUS RASHFORD")
                                                    )]
    dataset_neu = dataset["text"][
        (dataset["vader_emotion"] == "neutral") & ((dataset["player_name"] == "HARRY MAGUIRE") |
                                                   (dataset["player_name"] == "ALEXANDER ISAK") |
                                                   (dataset["player_name"] == "ERLING BRAUT HAALAND") |
                                                   (dataset["player_name"] == "DARWIN NUNEZ") |
                                                   (dataset["player_name"] == "MYKHAYLO MUDRYK") |
                                                   (dataset["player_name"] == "KAORU MITOMA") |
                                                   (dataset["player_name"] == "BUKAYO SAKA") |
                                                   (dataset["player_name"] == "JAMES MADDISON") |
                                                   (dataset["player_name"] == "HARRY KANE") |
                                                   (dataset["player_name"] == "MARCUS RASHFORD")
                                                   )]

    datasets = (('NEGATIVE', dataset_neg), ('POSITIVE', dataset_pos), ('NEUTRAL', dataset_neu))

    print("Numero Tweet neutrali:",len(dataset_neu),
          "\nNumero Tweet positivi:",len(dataset_pos),
           "\nNumero Tweet negativi:", len(dataset_neg))
    # PERFORMS TOPIC MODELLING
    topic_modelling(datasets)
