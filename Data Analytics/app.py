# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from collections import Counter
from PIL import Image
from zlib import DEF_BUF_SIZE
from click import style
import numpy as np
from dash import Dash, dcc, html,dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input,Output,State
from nltk.tokenize import word_tokenize
import pandas as pd
import os
from flask import Flask
import preprocess
import lexicon
import ml
import pickle
import plotly.express as px
import plotly.graph_objs as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt # plotting
from plotly.graph_objects import Layout

server = Flask(__name__)


THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
NAIVE_MODEL_UNI=os.path.join(THIS_FOLDER,'model_pretrained/model_naive_uni.sav')
NAIVE_MODEL_BI=os.path.join(THIS_FOLDER,'model_pretrained/model_naive_bi.sav')
NAIVE_MODEL_TFIDF=os.path.join(THIS_FOLDER,'model_pretrained/model_naive_tfidf.sav')
LOGISTIC_MODEL_UNI=os.path.join(THIS_FOLDER,'model_pretrained/model_logistic_uni.sav')
LOGISTIC_MODEL_BI=os.path.join(THIS_FOLDER,'model_pretrained/model_logistic_bi.sav')
LOGISTIC_MODEL_TFIDF=os.path.join(THIS_FOLDER,'model_pretrained/model_logistic_tfidf.sav')
YELP_LEXICON=os.path.join(THIS_FOLDER,'dataset/lexicon_yelp_review.csv')


colors = {
    'background': '#111111',
    'background2': '#FF0',
    'text': 'yellow'
    }
CONTENT_STYLE = {
     'margin-left': '25%',
     'margin-right': '5%',
    'padding': '20px 10p'
}

TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#191970'
}

CARD_TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#0074D9'
}
SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '20%',
    'padding': '20px 10px',
    'background-color': '#f8f9fa'
}

csv_file=os.path.join(THIS_FOLDER, 'food.tsv')
preprocessed_csv=os.path.join(THIS_FOLDER,'dataset/preprocessed_dataset.csv')
try:
    df=pd.read_csv(preprocessed_csv, sep=',')
except FileNotFoundError:
    df = pd.read_csv(csv_file, sep='\t',encoding='cp1252')

    df.loc[df['score'] <=3, 'score'] = 0
    df.loc[df['score'] >=4, 'score'] = 1

    df['clean_text']=preprocess.clean_data(df['text'])

    token=preprocess.tokenize(df)

    token_stop=preprocess.stopword(token)

    token_stop_punct=preprocess.remove_punct(token_stop)

    token_stop_punct_lemma=preprocess.lemmatizer(token_stop_punct)

    df['token']=token_stop_punct_lemma
    df['join_token']=ml.jointoken(token_stop_punct_lemma)
    df_lexicon=lexicon.predict_lexicon(df)
    df_lexicon.to_csv(YELP_LEXICON,sep=',')
    df.to_csv(preprocessed_csv,sep=',')
# df_afinn=lexicon.affin(df)
df_lexicon= pd.read_csv(YELP_LEXICON,sep=',')
#lexicon.scoreYelp(df)

# df_lexicon['myscore_nlc']=lexicon.score_NLC(token_stop_punct_lemma)

x = df['join_token'].values.astype('U')
y = df['score']
X_train,X_test,y_test,y_train=ml.train_test(x,y)
X_train_uni,X_test_uni=ml.bow_unigram(X_train,X_test)
X_train_tfidf,X_test_tfidf=ml.tfid_vector(X_train,X_test)
X_train_bi,X_test_bi=ml.bow_bigram(X_train,X_test)
try:
    naive_model_uni = pickle.load(open(NAIVE_MODEL_UNI, "rb"))
    print("loaded {}".format(NAIVE_MODEL_UNI))
except (OSError, IOError) as e:
        ml.plot_roc(X_train_uni,y_train,X_test_uni,y_test,filename=NAIVE_MODEL_UNI)
        naive_model_uni = pickle.load(open(NAIVE_MODEL_UNI, "rb"))
try:
    naive_model_bi = pickle.load(open(NAIVE_MODEL_BI, "rb"))
    print("loaded {}".format(NAIVE_MODEL_BI))

except (OSError, IOError) as e:
        ml.plot_roc(X_train_bi,y_train,X_test_bi,y_test,filename=NAIVE_MODEL_BI)
        naive_model_bi = pickle.load(open(NAIVE_MODEL_BI, "rb"))
try:
    naive_model_tfidf = pickle.load(open(NAIVE_MODEL_TFIDF, "rb"))
    print("loaded {}".format(NAIVE_MODEL_TFIDF))

except (OSError, IOError) as e:
        ml.plot_roc(X_train_tfidf,y_train,X_test_tfidf,y_test,filename=NAIVE_MODEL_TFIDF)
        naive_model_tfidf = pickle.load(open(NAIVE_MODEL_TFIDF, "rb"))
try:
    logistic_model_uni = pickle.load(open(LOGISTIC_MODEL_UNI, "rb"))
    print("loaded {}".format(LOGISTIC_MODEL_UNI))

except (OSError, IOError) as e:
        C,penalty,solver=ml.optimal_c(X_train_uni,y_train)
        ml.train_logistic(X_train_uni,y_train,X_test_uni,y_test,C=C,solver=solver,penalty=penalty,filename=LOGISTIC_MODEL_UNI)
        logistic_model_uni = pickle.load(open(LOGISTIC_MODEL_UNI, "rb"))
try:
    logistic_model_bi = pickle.load(open(LOGISTIC_MODEL_BI, "rb"))
    print("loaded {}".format(LOGISTIC_MODEL_BI))
except (OSError, IOError) as e:
        C,penalty,solver=ml.optimal_c(X_train_bi,y_train)
        print(C)
        ml.train_logistic(X_train_bi,y_train,X_test_bi,y_test,C=C,solver=solver,penalty=penalty,filename=LOGISTIC_MODEL_BI)
        logistic_model_bi = pickle.load(open(LOGISTIC_MODEL_BI, "rb"))
try:
    logistic_model_tfidf = pickle.load(open(LOGISTIC_MODEL_TFIDF, "rb"))
    print("loaded {}".format(LOGISTIC_MODEL_TFIDF))
except (OSError, IOError) as e:
        C,penalty,solver=ml.optimal_c(X_train_tfidf,y_train)
        ml.train_logistic(X_train_tfidf,y_train,X_test_tfidf,y_test,C=C,solver=solver,penalty=penalty,filename=LOGISTIC_MODEL_TFIDF)
        logistic_model_tfidf = pickle.load(open(LOGISTIC_MODEL_TFIDF, "rb"))
df_ml=ml.predict_data(X_test_bi,y_test,logistic_model_bi,df)
app = Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
app.title='Amazon Food Review Sentiment Analysis'
app.config.suppress_callback_exceptions=True
header_text=html.Div('Amazon Food Review Analysis Dashboard',id='main_header_text',className='main-header',
                     style=dict(color='#1dabdd',
                     fontWeight='bold',width='100%',paddingTop='1vh',
                     display= 'flex', alignItems= 'center', justifyContent= 'center'))





controls = dbc.Row(dbc.Col(
    [   
        
        html.P('Engine', style={
            'textAlign': 'center'
        }),
        dcc.Dropdown(
            id='dropdown_engine',
            options=['ML', 'LEXICON',
            ],value='ML',
            multi=False,
            clearable=False,
            placeholder="Select Engine",
        ),
        html.Br(),
     
        dbc.Button(
            id='set_engine',
            n_clicks=0,
            children='Set Engine',
            color='primary',
            #block=True
        ),
        html.H2('Filter', style=TEXT_STYLE),
        html.Hr(),
        html.P('Product_ID', style={
            'textAlign': 'center'
        }),
        dcc.Dropdown(
            id='dropdown',
            multi=False,
            placeholder="Select a Product ID",
        ),
        html.Br(),
     
        dbc.Button(
            id='submit_button',
            n_clicks=0,
            children='Submit',
            color='primary',
            #block=True
        ),
        html.H2('Info', style=TEXT_STYLE),
        html.Hr(),
        html.H4(id='card_review_1', children=['Most review product'], className='card-title',
                                style=CARD_TEXT_STYLE),
        html.P(id='most_review', style=CARD_TEXT_STYLE),
        html.Hr(),
        html.H4(id='card_review_2', children=['Most active user'], className='card-title',
                                style=CARD_TEXT_STYLE),
        html.P(id='most_active_user', style=CARD_TEXT_STYLE),


    ]
)
)


sidebar = html.Div(
    [
        
        controls
    ],
    style=SIDEBAR_STYLE,
)
content_first_row = dbc.Row([
    dbc.Col(
        dbc.Card(
            [

                dbc.CardBody(
                    [
                        html.H4(id='card_title_1', children=['Number of review'], className='card-title',
                                style=CARD_TEXT_STYLE),
                        html.P(id='num_review', style=CARD_TEXT_STYLE),
                    ]
                )
            ]
        ),
        md=3
    ),
    dbc.Col(
        dbc.Card(
            [

                dbc.CardBody(
                    [
                        html.H4('Number of Positive Review', className='card-title', style=CARD_TEXT_STYLE),
                        html.P(id='pos_review', style=CARD_TEXT_STYLE),
                    ]
                ),
            ]

        ),
        md=3
    ),
    dbc.Col(
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H4('Number of negative Review', className='card-title', style=CARD_TEXT_STYLE),
                        html.P(id="neg_review", style=CARD_TEXT_STYLE),
                    ]
                ),
            ]

        ),
        md=3
    ),
    dbc.Col(
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H4('Accuracy', className='card-title', style=CARD_TEXT_STYLE),
                        html.P(id='accuracy', style=CARD_TEXT_STYLE),
                    ]
                ),
            ]
        ),
        md=3
    )
])

content_second_row=dbc.Row(
    [
        dbc.Col(
            dcc.Graph(
                id='predict_graph'
            )
           
        )
    ]
)

content_third_row=dbc.Row(
    [
        dbc.Col(
            id='slider'
        )
    ]
)
content_fourth_row=dbc.Row(
    [
        dbc.Col(
            id='data_table'
        )
    ]
)
content_fifth_row=dbc.Row(
    [
       dbc.Col(
               dcc.Graph(id='word_cloud')
            ),
            dbc.Col(
                dcc.Graph(id='pie_word')
            )
        
    ]
)
content=html.Div([
    html.H2('Analytics Food Review Dashboard'),
    html.Hr(),
    content_first_row,
    content_second_row,
    content_third_row,
    html.Hr(),
    content_fourth_row,
    html.Hr(),
    content_fifth_row,
],style=CONTENT_STYLE)

app.layout=html.Div([sidebar,content])

@app.callback(
    [Output('pie_word', 'figure')],
[Input('set_engine', 'n_clicks'),Input('submit_button', 'n_clicks')],
[State('dropdown_engine','value'),State('dropdown','value')]
)
def create_pie_chart(n_click,n,dropdown_engine,dropdown):
    df_predict=pd.DataFrame()
    if dropdown_engine=='ML':
        layout = Layout(plot_bgcolor='rgba(0,0,0,0)')
        fig = go.Figure(data=[go.Scatter(x=[], y=[])],layout=layout)
        fig.update_xaxes(visible=False,linecolor='white', gridcolor='white')
        fig.update_yaxes(visible=False,linecolor='white', gridcolor='white')
        return [fig]
    else:
        df_predict=df_lexicon
        if dropdown is None:
            values=[df_predict['n_pos'].sum(),df_predict['n_neg'].sum(),df_predict['n_neut'].sum()]
            name=['positive','negative','neutral']
            fig = px.pie(df_predict, values=values, names=name, title='Polarity Word Pie Chart')

            return [fig]
        else:
            filters=df_predict['product_id']=='{}'.format(dropdown)
            df_graph = df_predict.where(filters ,inplace=False) 
            df_graph=df_graph.dropna()
            values=[df_graph['n_pos'].sum(),df_graph['n_neg'].sum(),df_graph['n_neut'].sum()]
            name=['positive','negative','neutral']
            fig = px.pie(df_graph, values=values, names=name, title='Polarity Word Pie Chart')

            return [fig]

@app.callback(
[Output('dropdown', 'options')],
[Input('set_engine', 'n_clicks')],
[State('dropdown_engine','value')]
)
def update_dropdown_product(n_click,dropdown):
    df_predict=pd.DataFrame()
    if dropdown=='ML':
        df_predict=df_ml
    else:
        df_predict=df_lexicon
    options=[{'label':i,'value':i} for i in df_predict['product_id'].unique()]
    return [options]

@app.callback(
    [Output('most_review', 'children'),Output('most_active_user', 'children')],
    [Input('submit_button', 'n_clicks'),[Input('set_engine', 'n_clicks')]],
     [State('dropdown_engine','value')]
)
def update_info(n_clicks,n_click,dropdown_engine,):
    df_predict=pd.DataFrame()
    if dropdown_engine=='ML':
        df_predict=df_ml
    else:
        df_predict=df_lexicon
    return[df_predict.product_id.mode(),df_predict.user_id.mode()]

@app.callback(
    [Output('word_cloud', 'figure')],
    [Input('submit_button', 'n_clicks'),[Input('set_engine', 'n_clicks')]],
     [State('dropdown', 'value'),State('dropdown_engine','value')]
)
def create_wordcloud(n_clicks,n_click,dropdown,dropdown_engine):
    df_predict=pd.DataFrame()
    if dropdown_engine=='ML':
        df_predict=df_ml
    else:
        df_predict=df_lexicon
    if dropdown is None:
        data = df_predict['token'].replace("'", '', regex=True)
        data = data.replace("[", '' )
        data = data.replace("]", '' )
        mask = np.array(Image.open("assets/amazon.png"))
        my_wordcloud =WordCloud( background_color="white",mask=mask,contour_color='#F8981D',contour_width=3).generate(''.join(data))

        fig_wordcloud = px.imshow(my_wordcloud, template='ggplot2',
                                title="WordCloud")
        fig_wordcloud.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        fig_wordcloud.update_xaxes(visible=False)
        fig_wordcloud.update_yaxes(visible=False)

        return [fig_wordcloud]
    else:
        filters=df_predict['product_id']=='{}'.format(dropdown)
        df_graph = df_predict.where(filters ,inplace=False) 
        df_graph=df_graph.dropna()
        data = df_graph['token'].replace("'", '', regex=True)
        data = data.replace("[", '' )
        data = data.replace("]", '' )
        mask = np.array(Image.open("assets/amazon.png"))
        my_wordcloud =WordCloud( background_color="white",mask=mask,contour_color='#F8981D',contour_width=3).generate(''.join(data))

        fig_wordcloud = px.imshow(my_wordcloud, template='ggplot2',
                                title="WordCloud")
        fig_wordcloud.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        fig_wordcloud.update_xaxes(visible=False)
        fig_wordcloud.update_yaxes(visible=False)

        return [fig_wordcloud]



@app.callback(
      [Output('accuracy', 'children')],
    [Input('submit_button', 'n_clicks'),[Input('set_engine', 'n_clicks')]],
    [State('dropdown', 'value'),State('dropdown_engine','value')]
)
def accuracy(n_clicks,n_click,dropdown,dropdown_engine):
    df_predict=pd.DataFrame()
    if dropdown_engine=='ML':
        df_predict=df_ml
    else:
        df_predict=df_lexicon
    if dropdown is None:
        confusion = pd.crosstab(np.sign(df_predict.score), 
                        np.sign(df_predict.predict))
        confusion
        accuracy = np.sum(np.diag(confusion)) / np.sum(confusion.values)
        accuracy=accuracy*100
        text='{:.2f}{}'.format(accuracy,"%")
        return [text]
    else:
        filters=df_predict['product_id']=='{}'.format(dropdown)
        df_graph = df_predict.where(filters ,inplace=False) 
        df_graph=df_graph.dropna()
        confusion = pd.crosstab(np.sign(df_graph.score), 
                        np.sign(df_graph.predict))
        confusion
        accuracy = np.sum(np.diag(confusion)) / np.sum(confusion.values)
        accuracy=accuracy*100
        text='{:.2f}{}'.format(accuracy,"%")
        return [text]

@app.callback(
    [Output('slider', 'children')],
    [Input('submit_button', 'n_clicks'),[Input('set_engine', 'n_clicks')]],
    [State('dropdown', 'value'),State('dropdown_engine','value')]
)
def update_slider(n_clicks,n_click,dropdown_value,dropdown):
    df_predict=pd.DataFrame()
    if dropdown=='ML':
        df_predict=df_ml
    else:
        df_predict=df_lexicon
    if dropdown_value is None:
         num_review=len(df_predict)#.v
    else:   
        filters=df_predict['product_id']=='{}'.format(dropdown_value)
        df_graph = df_predict.where(filters ,inplace=False) 
        df_graph=df_graph.dropna()
        num_review=len(df_graph)#.value_counts()
    return [dcc.RangeSlider(0, num_review,1,
               value=[0,10],
               marks=None,
               id='my-slider',
               tooltip={"placement": "bottom", "always_visible": True}
    )]

@app.callback(
    [Output('num_review', 'children')],
    [Input('submit_button', 'n_clicks'),[Input('set_engine', 'n_clicks')]],
    [State('dropdown', 'value'),State('dropdown_engine','value')]) 
def update_num_review(n_clicks,n_click,dropdown_value,dropdown):
    df_predict=pd.DataFrame()
    if dropdown=='ML':
        df_predict=df_ml
    else:
        df_predict=df_lexicon
    if dropdown_value is None:
         num_review=len(df_predict)#.v
    else:   
        filters=df_predict['product_id']=='{}'.format(dropdown_value)
        df_graph = df_predict.where(filters ,inplace=False) 
     
        df_graph=df_graph.dropna()
        num_review=len(df_graph)#.value_counts()
    return [num_review]


@app.callback(
    [Output('pos_review', 'children')],
    [Input('submit_button', 'n_clicks'),[Input('set_engine', 'n_clicks')]],
    [State('dropdown', 'value'),State('dropdown_engine','value')]) 
def update_num_review(n_clicks,n_click,dropdown_value,dropdown):
    df_predict=pd.DataFrame()
    if dropdown=='ML':
        df_predict=df_ml
    else:
        df_predict=df_lexicon
    if dropdown_value is None:
         filters=df_predict['score']==1
         df_graph = df_predict.where(filters,inplace=False) 
         df_graph=df_graph.dropna()
         num_review=len(df_graph)#.v
    else:   
        filters=df_predict['product_id']=='{}'.format(dropdown_value)
        filters2=df_predict['score']==1
        df_graph = df_predict.where(filters & filters2,inplace=False) 
  
        df_graph=df_graph.dropna()
        num_review=len(df_graph)#.value_counts()
    return [num_review]

@app.callback(
    [Output('neg_review', 'children')],
    [Input('submit_button', 'n_clicks'),[Input('set_engine', 'n_clicks')]],
    [State('dropdown', 'value'),State('dropdown_engine','value')]) 
def update_num_review(n_clicks,n_click,dropdown_value,dropdown):
    df_predict=pd.DataFrame()
    if dropdown=='ML':
        df_predict=df_ml
    else:
        df_predict=df_lexicon
    if dropdown_value is None:
         filters=df_predict['score']==0
         df_graph = df_predict.where(filters,inplace=False) 
         df_graph=df_graph.dropna()
         num_review=len(df_graph)#.v
    else:   
        filters=df_predict['product_id']=='{}'.format(dropdown_value)
        filters2=df_predict['score']==0
        df_graph = df_predict.where(filters & filters2,inplace=False) 
  
        df_graph=df_graph.dropna()
        num_review=len(df_graph)#.value_counts()
    return [num_review]

    
@app.callback(
    [Output('predict_graph', 'figure')],
    [Input('submit_button', 'n_clicks'),Input('my-slider','value'),[Input('set_engine', 'n_clicks')]],
    [State('dropdown', 'value'),State('dropdown_engine','value')])
def update_predict_graph(n_clicks, slider_value,n_click,dropdown_value,dropdown):
    df_predict=pd.DataFrame()
    if dropdown=='ML':
        df_predict=df_ml
    else:
        df_predict=df_lexicon
    
    if dropdown_value is None:
        fig=px.line(df_predict.iloc[slider_value[0]:slider_value[1]], x="user_id", y=["score","predict"],markers=True)
    else:
     
        filters=df_predict['product_id']=='{}'.format(dropdown_value)
        df_graph = df_predict.where(filters,inplace=False) 

        df_graph=df_graph.dropna()
        
        fig=px.line(df_graph.iloc[slider_value[0]:slider_value[1]], x="user_id", y=["score","predict"],markers=True)
    return [fig]

@app.callback(
   [Output('data_table', 'children')], 
    [Input('submit_button', 'n_clicks'),[Input('set_engine', 'n_clicks')]],
    [State('dropdown', 'value'),State('dropdown_engine','value')]
)
def update_table(n_clicks,n_click,dropdown_value,dropdown):
    df_predict=pd.DataFrame()
    if dropdown=='ML':
        df_predict=df_ml
    else:
        df_predict=df_lexicon
    if dropdown_value is None:
        df_table=df_predict[['product_id','user_id','text','score','predict']]
        return [dash_table.DataTable(
           
            data=df_table.to_dict('records'),
            columns=[{'id': c, 'name': c} for c in df_table.columns],
            page_size=10,
   
            tooltip_data=[
                {
                    column:{'value':str(value),'type':'markdown'}
                    for column, value in row.items()
                }for row in df_table.to_dict('records')
            ],
              style_cell={
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
                'maxWidth': 0,
                },
            tooltip_delay=0,
            tooltip_duration=None
)]
    else:
        filters=df_predict['product_id']=='{}'.format(dropdown_value)
        df_graph = df_predict.where(filters,inplace=False) 

        df_graph=df_graph.dropna()
        df_table=df_graph[['product_id','user_id','text','score','predict']]
        return [dash_table.DataTable(
           
            data=df_table.to_dict('records'),
            columns=[{'id': c, 'name': c} for c in df_table.columns],
            page_size=10,
   
            tooltip_data=[
                {
                    column:{'value':str(value),'type':'markdown'}
                    for column, value in row.items()
                }for row in df_table.to_dict('records')
            ],
              style_cell={
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
                'maxWidth': 0,
                },
            tooltip_delay=0,
            tooltip_duration=None
)]
        
       

if __name__ == '__main__':
    app.run_server(debug=False)
