from dash import Dash, dcc, html, Input, Output
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pickle

if __name__ == "__main__":
    #import models and scaler
    linModelFile = 'linearModel.pickle'
    with open(linModelFile, 'rb') as file:
        linearModel = pickle.load(file)
    scalerFile = 'linearScaler.pickle'
    with open(scalerFile, 'rb') as file1:
        scaler = pickle.load(file1)
    dfxFile = 'dfx.pickle'
    with open(dfxFile, 'rb') as file2:
        dfx = pickle.load(file2)
    dfyFile = 'dfy.pickle'
    with open(dfyFile, 'rb') as file3:
        dfy = pickle.load(file3)
    rfFile = 'rf.pickle'
    with open(rfFile, 'rb') as file4:
        randomForest = pickle.load(file4)
    ridgeFile = 'ridge.pickle'
    with open(ridgeFile, 'rb') as file4:
        ridgeModel = pickle.load(file4)

    #prepare predictions
    dfx = scaler.transform(dfx)
    y_pred = linearModel.predict(dfx)
    dfy['y_pred'] = y_pred

    #create figure using predictions
    fig = px.line(dfy, x=dfy.index, y='y_pred', title='Projected Sales')
    
    #instantiate dash application
    app = Dash(__name__)
    app.layout = html.Div(
        children = [
            html.H1(children="Mansfield Sales Projections",
            ),
            dcc.Dropdown(['Linear', 'Ridge', 'Random Forest'], 'Linear', id='model-dropdown'),
            html.Div(id='model-output-container'),
            dcc.Graph(id='graph-output'),
        ]
    )
    @app.callback(
        Output(component_id='graph-output', component_property='figure'),
        Input(component_id='model-dropdown', component_property='value'),
    )
    def update_figure(model_selected):
        #predict sales using model selected
        print(model_selected)
        if model_selected == 'Linear':
            dfy['y_pred'] = linearModel.predict(dfx)
        elif model_selected == 'Ridge':
            dfy['y_pred'] = ridgeModel.predict(dfx)
        else:
            dfy['y_pred'] = randomForest.predict(dfx)

        #plot new predictions
        figure = px.line(dfy, x=dfy.index, y='y_pred')
        return(figure)

    app.run_server()
