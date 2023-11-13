import streamlit as st
from PIL import Image
import requests
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import imdb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
from collections import Counter
import re

img = Image.open('logo.png')

path1 = "./Animation - GitHub.json"
path2 = "./Animation - watch_movie.json"
movies_reduced = pd.read_csv("movies_stream_app.csv")
movies_cbf = pd.read_csv("movies_cbf.csv")
df_genres = pd.read_csv("Genres.csv")



with open(path1,"r") as file:
    url1 = json.load(file)

with open(path2,"r") as file:
    url2 = json.load(file)



st.set_page_config(
        page_title="Mon APP",
        page_icon= img,
        layout='wide',
        initial_sidebar_state="expanded")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: visible;}
            footer:after{
                background-color:#27dce0;
                font-size:14px;
                font-weight:6px;
                height:30px;
                margin:1rem;
                padding:0.8rem;
                content:'Travail réalisé par KERAN, SKANDER, KHALED, KEMAYOU  ';
                display: flex;
                align-items:center;
                justify-content:center;
                color:#4628dd;
            }
            header {visibility: hidden;}
            </style>
            """

st.markdown(hide_st_style, unsafe_allow_html=True)


with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        st.write("")
        st.title('SYSTÈME DE RECOMMANDATION DE FILMS')
    with right_column:
        st_lottie(url2,
                  reverse=True,
                  height=300,
                  width=300,
                  speed=0.5,
                  loop=True,
                  quality='high',
                  key='Movie'
                  )

with st.sidebar:
    selected = option_menu(
                menu_title="RECOMMANDATIONS DE FILMS",
                options=["Accueil", "Exploration de données", "Modélisation", "Application","Github-Repo"],
                icons=["house", "search","cpu", "terminal-plus","github"],
                menu_icon="three-dots",
                default_index=0,
                styles={
                "container": {"padding": "5!important", "background-color": "#0E1117" , "Font-family":"Monospace"},
                "icon": {"color": "#4628dd", "font-size": "21px"},
                "nav-link": {"font-size": "18px", "text-align": "left", "margin":"0px","Font-family":"Monospace"},
                "nav-link-selected": {"background-color": "#27dce0"},
                }
                )
    if selected == "Github-Repo":
        st_lottie(url1,
                  reverse=True,
                  height=150,
                  width=150,
                  speed=0.5,
                  loop=True,
                  quality='high',
                  key='Git'
                  )
        st.subheader("Consultez notre code ici :arrow_down::arrow_down: : ")
        st.markdown(
            """
            <div style='
            background-color: #0E1117; 
            cursor:pointer; 
            height:2.8rem;
            font-size:18px;
            font-weight:bolder;
            border-radius:5px;
            font-family: Monospace;
            box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2), 0 6px 20px 0 rgba(0,0,0,0.19);
            display: flex;
            align-items:center;
            justify-content:center;'>
                    <a  href="https://github.com/DataScientest-Studio/aug23_bds_reco_films" 
                    style='color: white; 
                           text-decoration:none;
                           padding-top:6px;
                           padding-bottom:5px;
                           text-align:center;'>
                    GITHUB
                    </a>
            </div>
            """,
            unsafe_allow_html=True,
        )
if selected == "Accueil":
    st.header("INTRODUCTION")
    st.write(
        "Un système de recommandation est une forme spécifique de filtrage de l'information qui a pour but de présenter à un utilisateur des éléments qui sont susceptibles de l'intéresser, et ce, en se basant sur ses préférences et son comportement.")
    st.image(Image.open(
        requests.get("https://dropinblog.net/34242773/files/Systeme_de_recommandation.png", stream=True).raw))
    st.write(
        "Notre projet consiste à créer un système de recommandation pour prédire la 'préfrence' pour un utilisateur")
    st.write(
        "Pour ce faire, nous allons créer modèle de ML capable de prédire et de recommander des films aux autres utilisateurs en fonction de leurs préférences et de leurs besoins.")
    st.write("Les jeux des données utilisés sont issus des bases ci-dessous:")
    st.write("https://grouplens.org/datasets/movielens/20m/")
    st.write("https://www.imdb.com/interfaces/")
if selected == "Exploration de données":
    st.header("EXPLORATION DE DONNÉES")
    st.write("Les données récoltées contiennent 26744 films et plus de 2 millions de notations.")
    st.subheader(":one: Les catégories de films")
    st.write("Plusieurs films possèdent plus qu’un genre, il était donc nécessaire d’attribuer un seul genre à chaque film. ")
    st.write("Les différents genres des films sont représentés dans lle tableau ci-dessous ")
    st.dataframe(df_genres.iloc[: , 1:])
    st.write("On remarque que dans toute la banque de films, le genre ‘Drama’ est le plus représenté, suivi des ‘Comedy’")

    st.bar_chart(data= df_genres, x="genre", y='count', use_container_width=True)
    st.write("Voici les meilleurs genres présents dans la base de données")
    number = st.number_input("Choisir un nombre ", min_value=1, max_value=20, value=5, step=1, placeholder="Entrez un nombre...")

    specs = [[{'type': 'pie'}, {"type": "bar"}]]
    fig = make_subplots(rows=1, cols=2, specs=specs, shared_yaxes=True, subplot_titles=['Graphique en secteurs (Camembert)',
                                                                                        'Diagramme en bar'])
    fig.add_trace(go.Pie(
        labels=df_genres.head(number)['genre'],
        values=df_genres.head(number)['count'],
        hole=0.6,
        marker_colors=['#353837', '#646665', '#8e9492', '#c9d1ce'],
        textinfo='percent+value',
    ), 1, 1)

    fig.add_trace(go.Bar(
        x=df_genres.head(number)['genre'],
        y=df_genres.head(number)['count'],
        base=df_genres.head(number),
    ), 1, 2)

    fig.update_layout(showlegend=False,
                      title=dict(text="VISUALISATION DES {} GENRES LES PLUS POPULAIRES".format(number),
                                 font=dict(
                                     family="Monospace",
                                     size=20,
                                     color='#283747')
                                 ))
    st.write(fig)


    st.write("Afin de savoir s'il existe une corrélation entre les différents genres nous avons tracé la matrice de corrélations de notre DataFrame ")

    fig = px.imshow(movies_cbf.iloc[:,:20].corr(), text_auto=True)
    st.write(fig, width=300)

    st.write("Les seules corrélations présentes se trouvent entre les variables ‘Animation’ et ‘Children’, ce qui semble cohérent.")
    st.write("On peut également noter quelques corrélations entre ‘Crime’, ‘Thriller’, et ‘Mystery’ voire ‘Horror’, mais elles sont limitées.")
    st.subheader(":two: Les notations de films")
    st.write("Il y’a 27262 films dont 26744 films notés c’est à dire environ 2% (518 films) n’ont pas de note")
    st.write("Nous avons effectué un pré-traitement des données de la base 'Rating', pour récupèrer la moyenne Bayesian qui prend en compte la popularité du film vu, autrement, nous gardant uniquement les films qui ont été noté suffisament et ensuite nous appliquons une moyenne aux notes données par chaque utilisateur")
    st.write("Ci-dessous la liste des 20 meilleurs films")

    st.dataframe(movies_reduced[['title','avg rating']])

    choice = st.slider('Choisissez le nombre de films à afficher ?', 1, 20, 5)

    fig2 = make_subplots(rows=2, cols=1, specs=[[{}], [{}]], shared_xaxes=False,
                        shared_yaxes=False, vertical_spacing=0.3, row_heights=[0.7, 0.3])
    fig2.add_trace(go.Bar(
        x=[rate*i for rate, i in zip(movies_reduced.head(choice)['avg rating'],list(np.log2(range(1, choice))))],
        y=movies_reduced.head(choice)['title'],
        text=[x for x in round(movies_reduced.head(choice)['avg rating'],2)],
        textposition='auto',
        marker=dict(
            color='#4628dd',
            line=dict(color='#4628dd',width=1),
        ),
        name='Les films les mieux notés',
        orientation='h',
    ), 1, 1)
    fig2.add_trace(go.Scatter(
        y=movies_reduced.head(choice)['number of Users watched'], x=movies_reduced.head(choice)['title'],
        mode='markers',
        marker=dict(
            color='LightSkyBlue',
            size=[5*i for i in reversed(range(1,choice+1))],
        ),
        name='Les films les plus vues',
    ), 2, 1)

    st.write(fig2)

    st.subheader(":three: Les commentaires")
    st.write('Nous avons exploré les commentaires liés à chaque film ')
    st.write('Ces données ont étés fournis dans le fichier "tags.csv" ')
    st.write("Ces données reflètent l'avis des personnes qui ont regardé le film ")
    st.write("Nous remarquons la présence de plusieurs commentaires positifs pour les films mieux notés")
    st.write("Nous pouvons réaliser un WordCloud afin de visualiser les 50 comentaires les plus employé pour chaque film")

    st.write('Veuillez sélectionner un films parmi les choix possibles')

    option_wc = st.radio("",['Pink Panther, The (1963)','Paris Is Burning (1990)','Black Mirror (2011)'])
    if option_wc:
        tags = movies_reduced[movies_reduced.title == option_wc]['tag'].to_list()
        tags = re.sub(r"[^\w\s]", "", tags[0])
        mots = re.findall(r'\w+', tags)
        wc = WordCloud(width=600, height=400, max_words=1000, random_state=1, background_color='white',
                   colormap='PuBu').generate_from_frequencies(dict(Counter(mots).most_common(50)))
        wc.to_file("image-wc.png")
        st.image("image-wc.png")
if selected == "Modélisation":
    data_model = pd.read_csv('data_model.csv')

    st.write("### Modélisation Machine Learning")
    data_model.index = ['RMSE', 'MSE', 'MAE', 'FCP', 'Run_time(s)']
    st.dataframe(data_model)
    colors = plt.cm.viridis(np.linspace(0, 1, data_model.columns.shape[0]))
    fig = plt.figure(figsize=(15, 15))
    plt.subplot(321)
    plt.bar(data_model.columns, data_model.iloc[0, :], color=colors)
    plt.xticks(rotation=60)
    plt.title("RMSE en fonction de l'algorithme")
    plt.subplot(322)
    plt.bar(data_model.columns, data_model.iloc[1, :], color=colors)
    plt.xticks(rotation=60)
    plt.title("MSE en fonction de l'algorithme")

    plt.subplots_adjust(hspace=1)  # Ajuster l'espacement vertical

    plt.subplot(323)
    plt.bar(data_model.columns, data_model.iloc[2, :], color=colors)
    plt.xticks(rotation=60)
    plt.title("MAE en fonction de l'algorithme")
    plt.subplot(324)
    plt.bar(data_model.columns, data_model.iloc[3, :], color=colors)
    plt.xticks(rotation=60)
    plt.title("FCP en fonction de l'algorithme")

    plt.subplots_adjust(hspace=0.5)  # Ajuster l'espacement vertical

    plt.subplot(325)
    plt.bar(data_model.columns, data_model.iloc[4, :], color=colors)
    plt.xticks(rotation=60)
    plt.title("Run_time(s) en fonction de l'algorithme")

    st.pyplot(fig)

    choix = data_model.columns
    option = st.selectbox('Choix du modèle', choix)
    st.write('Le modèle choisi est :', option)
    # clf = prediction(option)
    display = st.radio('Que souhaitez-vous montrer ?', ('RMSE', 'MSE', 'MAE', 'FCP', 'Run_time(s)'))
    if display == 'RMSE':
        st.latex(r'RMSE = \sqrt{\frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{n}}')
        st.write('Mesure la racine carrée de la moyenne des carrés des erreurs entre les valeurs réelles et les prédictions. Plus la valeur de RMSE est basse, meilleure est la performance du modèle. RMSE donne une indication de la magnitude des erreurs.')
        st.write(data_model.loc['RMSE', option])
    elif display == 'MSE':
        st.latex(r'MSE = \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{n}')
        st.write('Mesure la moyenne des carrés des erreurs entre les valeurs réelles et les prédictions. Comme le carré accentue les erreurs importantes, la MSE peut être sensible aux valeurs aberrantes.')
        st.write(data_model.loc['MSE', option])
    elif display == 'MAE':
        st.latex(r'MAE = \frac{\sum_{i=1}^{n}|y_i - \hat{y}_i|}{n}')
        st.write('Mesure la moyenne des valeurs absolues des erreurs entre les valeurs réelles et les prédictions. Contrairement à la MSE, la MAE n\'accorde pas de poids excessif aux erreurs importantes.')
        st.write(data_model.loc['MAE', option])
    elif display == 'FCP':
        st.latex(r'FCP = \frac{C}{C + D}')
        st.write('Utilisé pour évaluer les systèmes de recommandation basés sur des classements ordinaux. Il mesure la fraction de paires pour lesquelles le classement du modèle est conforme au classement réel.')
        st.write(data_model.loc['FCP', option])

    elif display == 'Run_time(s)':
        st.write('Temps d\'exécution du modèle : ')
        st.write(data_model.loc['RMSE', option])
    elif display == 'MSE':
        st.latex(r'MSE = \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{n}')
        st.write('Mesure la moyenne des carrés des erreurs entre les valeurs réelles et les prédictions. Comme le carré accentue les erreurs importantes, la MSE peut être sensible aux valeurs aberrantes.')
        st.write(data_model.loc['MSE', option])
    elif display == 'MAE':
        st.latex(r'MAE = \frac{\sum_{i=1}^{n}|y_i - \hat{y}_i|}{n}')
        st.write('Mesure la moyenne des valeurs absolues des erreurs entre les valeurs réelles et les prédictions. Contrairement à la MSE, la MAE n\'accorde pas de poids excessif aux erreurs importantes.')
        st.write(data_model.loc['MAE', option])
    elif display == 'FCP':
        st.latex(r'FCP = \frac{C}{C + D}')
        st.write('Utilisé pour évaluer les systèmes de recommandation basés sur des classements ordinaux. Il mesure la fraction de paires pour lesquelles le classement du modèle est conforme au classement réel.')
        st.write(data_model.loc['FCP', option])

    elif display == 'Run_time(s)':
        st.write('Temps d\'exécution du modèle : ')
        st.write(data_model.loc['Run_time(s)', option])

    st.write('---------------------------------------------------------------------------------------------------------')
    # Affichez une checkbox pour montrer ou cacher la première case
    lis_genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime','Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX',
                'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', '(no genres listed)']
    show_box1 = st.checkbox(lis_genres[0])
    show_box2 = st.checkbox(lis_genres[1])
    show_box3 = st.checkbox(lis_genres[2])
    show_box4 = st.checkbox(lis_genres[3])
    show_box5 = st.checkbox(lis_genres[4])
    show_box6 = st.checkbox(lis_genres[5])
    show_box7 = st.checkbox(lis_genres[6])
    show_box8 = st.checkbox(lis_genres[7])
    show_box9 = st.checkbox(lis_genres[8])
    show_box10 = st.checkbox(lis_genres[9])
    show_box11 = st.checkbox(lis_genres[10])
    show_box12 = st.checkbox(lis_genres[11])
    show_box13 = st.checkbox(lis_genres[12])
    show_box14 = st.checkbox(lis_genres[13])
    show_box15 = st.checkbox(lis_genres[14])
    show_box16 = st.checkbox(lis_genres[15])
    show_box17 = st.checkbox(lis_genres[16])
    show_box18 = st.checkbox(lis_genres[17])
    show_box19 = st.checkbox(lis_genres[18])
    show_box20 = st.checkbox(lis_genres[19])

    st.write('---------------------------------------------------------------------------------------------------------')

    Action = 0
    Adventure = 0
    Animation = 0
    Children = 0
    Comedy = 0
    Crime = 0
    Documentary = 0
    Drama = 0
    Fantasy = 0
    Film_Noir = 0
    Horror = 0
    IMAX = 0
    Musical = 0
    Mystery = 0
    Romance = 0
    Sci_Fi = 0
    Thriller = 0
    War = 0
    Western = 0
    no_genres_listed = 0

    # Affichez la première case conditionnellement
    if show_box1:
        Action = 1
    else :
        Action = 0
    if show_box2:
        Adventure = 1
    else :
        Adventure = 0
    if show_box3:
        Animation = 1
    else :
        Animation = 0
    if show_box4:
        Children = 1
    else :
        Children = 0
    if show_box5:
        Comedy = 1
    else :
        Comedy = 0

    if show_box6:
        Crime = 1
    else :
        Crime = 0
    if show_box7:
        Documentary = 1
    else :
        Documentary = 0
    if show_box8:
        Drama = 1
    else :
        Drama = 0


    if show_box9:
        Fantasy = 1
    else :
        Fantasy = 0

    if show_box10:
        Film_Noir = 1
    else :
        Film_Noir = 0

    if show_box11:
        Horror = 1
    else:
        Horror = 0
    if show_box12:
        IMAX = 1
    else :
        IMAX = 0
    if show_box13:
        Musical = 1
    else :
        Musical = 0
    if show_box14:
        Mystery = 1
    else :
        Mystery = 0
    if show_box15:
        Romance = 1
    else :
        Romance = 0
    if show_box16:
        Sci_Fi = 1
    else :
        Sci_Fi = 0
    if show_box17:
        Thriller = 1
    else :
        Thriller = 0

    if show_box18:
        War = 1
    else :
        War = 0
    if show_box19:
        Western = 1
    else :
        Western = 0

    if show_box20:
        no_genres_listed = 1
    else :
        no_genres_listed = 0

    new_line = pd.DataFrame([[Action, Adventure, Animation, Children, Comedy, Crime, Documentary, Drama, Fantasy, Film_Noir, Horror, IMAX,
                Musical, Mystery, Romance, Sci_Fi, Thriller, War, Western, no_genres_listed]], columns=['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime','Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX',
                'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', '(no genres listed)'])

    movies_cbf_user = pd.concat([movies_cbf.iloc[:,:20], new_line], ignore_index=True)
    afficher_matrice = st.checkbox('Cliquer ici pour afficher la matrice')

    if afficher_matrice:
        st.dataframe(movies_cbf_user)


    # Ajouter une case à remplir pour un entier
    entier_saisi = st.number_input("Entrez un entier", value=0, step=1)
    if entier_saisi < 0 :
        st.write('Entrez un entier supérieur ou égal à 0')
        entier_saisi = 0

    # Afficher la valeur saisie
    st.write(f"Vous avez saisi : {entier_saisi}")

    from sklearn.metrics.pairwise import cosine_similarity # La matrice cosinus permet d'évaluer le degré de similarité entre 2 vecteurs.

    cosine_sim = cosine_similarity(movies_cbf_user, movies_cbf_user)
    print(f"Dimensions of our genres cosine similarity matrix: {cosine_sim.shape}")

    n_recommendations = entier_saisi

    title = 'Visiteur'
    idx = 1500 # index du titre du film. A REMPLACER PAR UNE VARIABLE

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = list(filter(lambda x: x[0] != 1500, sim_scores)) # on enlève de la liste le tuple correspondant au film de base
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    similar_movies = [i[0] for i in sim_scores]


    st.write(f"Recommendations pour le {title}:")
    st.dataframe(movies_reduced['title'].iloc[similar_movies].head(entier_saisi), width = 700)

    # ATTENTION A LA RÉINDEXATION. IL FAUT S'ASSURER QUE CELA NE PERTURBE PAS LA SUITE DES OPÉRATIONS
    # Les films qui sortent sont classés, par construction, suivant leur moyenne bayesian


    st.write("### Modélisation Deep Learning")

    # Afficher l'image dans Streamlit
    st.image("model.summary.png", use_column_width=True)
    st.image("modele_deep_epochs.png", use_column_width=True)

    # Affichez une checkbox pour montrer ou cacher la première case

















if selected == "Application" :
    def get_recommendations(title):
        cosine_sim = cosine_similarity(movies_cbf, movies_cbf)
        movie_idx = dict(zip(movies_reduced['title'], list(movies_reduced.index)))
        n_recommendations = 5

        idx = movie_idx[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:(n_recommendations+1)]
        similar_movies = [i[0] for i in sim_scores]

        return movies_reduced['title'].iloc[similar_movies].values.tolist()

    def fetch_poster(movies):
        url = []
        ids = []
        for element in movies :
            ids.append(movies_reduced[movies_reduced['title']==element]['imdbId'])

        for element in ids :
            url.append(imdb.IMDb().get_movie(element)['cover url'])
            
        return url

    st.write("Testons-nous notre model de recommandation !! 🍿🍿🍿")
    option = st.selectbox(
        'Quel film préférez-vous regarder', options = movies_reduced['title'], index=None, placeholder="Sélectionner un film ...")
    if st.button('Recommander 🚀') :
        with st.spinner('Recherche en cours ...'):
            movies_recommended = get_recommendations(option)
            posters = fetch_poster(movies_recommended)
        if posters:
            col1, col2, col3, col4, col5 = st.columns(5, gap='medium')
            with col1:
                st.image(posters[0])
                html_str = f"""                                     
                <span style="word-wrap:break-word;"></span>         
                <p class="a">{movies_recommended[0]}</p>            
                """
                col1.markdown(html_str, unsafe_allow_html=True)

            with col2:
                st.image(posters[1])
                html_str = f"""                                     
                <span style="word-wrap:break-word;"></span>         
                <p class="a">{movies_recommended[1]}</p>            
                """
                col2.markdown(html_str, unsafe_allow_html=True)

            with col3:
                st.image(posters[2])
                html_str = f"""                                 
                <span style="word-wrap:break-word;"></span>     
                <p class="a">{movies_recommended[2]}</p>        
                """
                col3.markdown(html_str, unsafe_allow_html=True)

            with col4:
                st.image(posters[3])
                html_str = f"""                                 
                <span style="word-wrap:break-word;"></span>     
                <p class="a">{movies_recommended[3]}</p>        
                """
                col4.markdown(html_str, unsafe_allow_html=True)

            with col5:
                 st.image(posters[4])
                 html_str = f"""
                 <span style="word-wrap:break-word;"></span>
                 <p class="a">{movies_recommended[4]}</p>
                 """
                 col5.markdown(html_str, unsafe_allow_html=True)