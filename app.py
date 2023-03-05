import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter('ignore')
import os
from dash import Input,State,Output,dcc,html,ctx,dash_table,Dash
from dash_bootstrap_templates import load_figure_template
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc 
import plotly.express as px
import plotly.offline as offline
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd 
from flask import Flask
import base64
import numpy as np
from sklearn import preprocessing
from utils.styling import style_app
from utils.read_data import read_data
dirname=os.path.dirname(__file__)

#-------------------Styling---------------------#
external_stylesheets,colors,min_style,color_scale=style_app()
figure_template=load_figure_template("sketchy")
image_path=os.path.join(dirname,os.path.normpath('utils/images/logo.png'))
image=base64.b64encode(open(image_path,'rb').read())
pio.templates.default = "sketchy+draft"

#app=Dash(__name__, external_stylesheets=external_stylesheets)

def create_table(df,id,renameable,pagesize=3):
    return html.Div(dash_table.DataTable(
        id=id,
        columns=[{"name": i, "id": i, "deletable": True,'renamable': renameable} for i in df.columns],
        data=df.to_dict("records"),
        page_size=pagesize,
        editable=True,
        row_deletable=True,
        filter_action="native",
        sort_action="native",
        sort_mode='multi',
        selected_rows=[],
        selected_columns=[],
        page_current=0,
        style_table={"overflowX": "auto"},
        row_selectable="multi"),className='dbc-row-selectable')
def cerate_Numeric(id,placeholder):
    return dbc.Input(id=id,type='Number',placeholder=placeholder,debounce=True,style=min_style)

def save_plot(fig,name,save_path):
    if save_path:
        path=os.path.join(save_path,name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
    else:
        path=name
    offline.plot(fig,filename=path,auto_open=True,include_plotlyjs='cdn')

def create_Tab1(df):
    dff=df.describe(include='all')
    dff.insert(0,'statistical values',dff.index)
    return dcc.Tab(label='Statistics',id='Col-tab',children=[dbc.Row(create_table(dff,'stats-table',False,pagesize=12)),
                                                             dbc.Row(dcc.Input(id='stats-name',type='text',placeholder='Name of the export',debounce=True,style=min_style)),
                                                             dbc.Row(html.Button('Export Statistics as csv',id='export-stats',style=min_style)),
                                                             dbc.Row(html.Div(id='stat-export',style=min_style))])    
def create_Tab2(df):
    columns=df.columns.to_list()
    return dcc.Tab(label='Histogram',id='Col-tab',children=[
    dbc.Row(dcc.Loading(id='Col-Loading',children=[dcc.Graph(id='Col-Graph',figure={})])),
    dbc.Row([html.H4('Plot Settings'),html.Hr()]),
    dbc.Row([html.H5('Column'),dcc.Dropdown(options=columns,id='Col-x-dropdown',placeholder='Select Column for Histogram',style=min_style),]),
    dbc.Row([html.H5('Color and Pattern'),dcc.Dropdown(options=columns,id='Col-color-dropdown',placeholder='Select Color Column',style=min_style),dcc.Dropdown(options=columns,id='Col-pattern-dropdown',placeholder='Select Pattern Column',style=min_style)]),
    dbc.Row([dcc.Input(id='Col-name',type='text',placeholder='Input Plot Title (This is also the file name when saving)',debounce=True,style=min_style),html.Button('Save Histogram Plot',id='Col-save-plot',style=min_style)])
        	])
def create_Tab3(df):
    num_columns=df.select_dtypes(include=np.number).columns.to_list()
    return dcc.Tab(label='Parallel Coordinates',id='PC-tab',children=[
    dbc.Row(dcc.Loading(id='PC-Loading',children=[dcc.Graph(id='PC-Graph',figure={})])),
    dbc.Row([html.H4('Plot Settings'),html.Hr()]),
    dbc.Row([html.H5('Color'),dcc.Dropdown(options=num_columns,id='PC-color-dropdown',placeholder='Select Color Column',style=min_style),]),
    dbc.Row([html.H5('Lower and Upper Bound'),cerate_Numeric('PC-Lower-Bound',placeholder='Lower Bound (without function)'),cerate_Numeric('PC-Upper-Bound',placeholder='Upper Bound (without function)')]),
    dbc.Row([dcc.Input(id='PC-name',type='text',placeholder='Input Plot Title (This is also the file name when saving)',debounce=True,style=min_style),html.Button('Save Parallel Coordinates Plot',id='PC-save-plot',style=min_style)])
        	])
def create_Tab4(df):
    num_columns=df.select_dtypes(include=np.number).columns.to_list()
    columns=df.columns
    return dcc.Tab(label='Scatterplot 2D',id='SC-tab',children=[
    dbc.Row(dcc.Loading(id='SC-Loading',children=[dcc.Graph(id='SC-Graph',figure={})])),
    dbc.Row([html.H4('Plot Settings'),html.Hr()]),
    dbc.Row([html.H5('Columns'),dcc.Dropdown(options=num_columns,id='SC-x-dropdown',placeholder='Select the x-Column',style=min_style),dcc.Dropdown(options=num_columns,id='SC-y-dropdown',placeholder='Select the y-Column',style=min_style)]),
    dbc.Row([html.H5('Color and Size'),dcc.Dropdown(options=columns,id='SC-color-dropdown',placeholder='Select Color Column',style=min_style),dcc.Dropdown(options=num_columns,id='SC-size-dropdown',placeholder='Select Size Column',style=min_style)]),
    dbc.Row([dcc.Input(id='SC-name',type='text',placeholder='Input Plot Title (This is also the file name when saving)',debounce=True,style=min_style),html.Button('Save Scatter Plot',id='SC-save-plot',style=min_style)])
        	])
def create_Tab5(df):
    num_columns=df.select_dtypes(include=np.number).columns.to_list()
    columns=df.columns
    return dcc.Tab(label='Scatterplot 3D',id='SC3D-tab',children=[
    dbc.Row(dcc.Loading(id='SC3D-Loading',children=[dcc.Graph(id='SC3D-Graph',figure={})])),
    dbc.Row([html.H4('Plot Settings'),html.Hr()]),
    dbc.Row([html.H5('Columns'),dcc.Dropdown(options=num_columns,id='SC3D-x-dropdown',placeholder='Select the x-Column',style=min_style),dcc.Dropdown(options=num_columns,id='SC3D-y-dropdown',placeholder='Select the y-Column',style=min_style),dcc.Dropdown(options=num_columns,id='SC3D-z-dropdown',placeholder='Select the z-Column',style=min_style)]),
    dbc.Row([html.H5('Color and Pattern'),dcc.Dropdown(options=columns,id='SC3D-color-dropdown',placeholder='Select Color Column',style=min_style),dcc.Dropdown(options=num_columns,id='SC3D-size-dropdown',placeholder='Select Size Column',style=min_style)]),
    dbc.Row([dcc.Input(id='SC3D-name',type='text',placeholder='Input Plot Title (This is also the file name when saving)',debounce=True,style=min_style),html.Button('Save 3D Scatter Plot',id='SC3D-save-plot',style=min_style)])
    ])


def create_Tab6(df):
    num_columns=df.select_dtypes(include=np.number).columns.to_list()
    columns=df.columns
    return dcc.Tab(label='Ridge',id='Ridge-tab',children=[
    dbc.Row(dcc.Loading(id='Ridge-Loading',children=[dcc.Graph(id='Ridge-Graph',figure={})])),
    dbc.Row([html.H4('Plot Settings'),html.Hr()]),
    dbc.Row([html.H5('Columns'),dcc.Dropdown(options=columns,id='Ridge-x-dropdown',placeholder='Select the x-Column',style=min_style),dcc.Dropdown(options=columns,id='Ridge-y-dropdown',placeholder='Select the y-Column',style=min_style)]),
    dbc.Row([html.H5('Color'),dcc.Dropdown(options=columns,id='Ridge-color-dropdown',placeholder='Select Color Column',style=min_style)]),
    dbc.Row([dcc.Input(id='Ridge-name',type='text',placeholder='Input Plot Title (This is also the file name when saving)',debounce=True,style=min_style),html.Button('Save 3D Scatter Plot',id='Ridge-save-plot',style=min_style)])
    ])

#Tab7=dcc.Tab(label='Pareto Analysis ABC Analyse',id='Pareto-tab',children=[html.H1('Test3')])

def create_Tab8(df):
    return dcc.Tab(label='Correlations',id='Corr-tab',children=[
    dbc.Row(dcc.Loading(id='Corr-Loading',children=[dcc.Graph(id='Corr-Graph',figure={})])),
    dbc.Row([html.H4('Plot Settings'),html.Hr()]),
    dbc.Row([html.H5('Correlation Type'),dcc.Dropdown(options=['pearson','spearman','kendall'],id='Corr-type-dropdown',placeholder='Select Correlation Type',style=min_style),]),
    dbc.Row([dcc.Input(id='Corr-name',type='text',placeholder='Input Plot Title (This is also the file name when saving)',debounce=True,style=min_style),html.Button('Save Correlations Plot',id='Corr-save-plot',style=min_style)])
    ])
def create_Export():
    return dcc.Tab(label='Export Data',id='Export-tab',children=[dbc.Row([dcc.Input(id='Export-name',type='text',placeholder='Name of the Data Export',debounce=True,style=min_style),html.Button('Export Data',id='Export Data',style=min_style),html.Div(id='Export-div')])])
#----------------------------------------------------------------------------

app=Dash(__name__,external_stylesheets=[dbc.themes.SKETCHY],suppress_callback_exceptions=True)
app.layout = dbc.Container([
                    #header
                    dbc.Row([
                            dbc.Col(html.H1(id='Header',children='Christoph`s Rapid Viz',className='Header')),html.Img(src=app.get_asset_url('logo.png'),style={'height':'60px','width':'105px'}),html.Hr()
                            ]),
                    #Table and GlobalSettings
                    dbc.Row([
                            dcc.Tabs(id='Table_Settings',children=[
                                    #TODO displaying data types
                                    dcc.Tab(label='Load Data and gernal setting',children=[
                                            dbc.Row([dbc.Col([dbc.Row(dcc.Input(id='Path',type='text',placeholder='Path to data (supportes *.xlsx,*.parquet,*.csv)',debounce=True,style=min_style)),dbc.Row(dcc.Input(id='Save_Path',type='text',placeholder='Path to where the plots shall be saved',debounce=True,style=min_style)),dbc.Row(html.Button('Load Data',id='Load-Data-button',n_clicks=0,style=min_style)),dbc.Row(dcc.Checklist(['Automatically convert datatypes'],['Automatically convert datatypes'],id='change_dtypes',style=min_style)),dbc.Row(html.Div(id='loading_info',style=min_style))]),
                                                    dbc.Col([dbc.Row(children=[dcc.Markdown('Welcome to Christoph´s Rapid Viz, a web based tool to visualize your Data! \n\n To start please insert the path of data you want to visualize and click the Button Load Data! \n\n PS: If you want to clear a dropdown, just use Backspace or Del',style={'text-align':'center'})]),
                                                            dbc.Row(html.Img(src=app.get_asset_url('pexels-anna-nekrashevich-6802049.jpg'),style={'height':'80%','width':'80%','display':'block','margin-left':'auto','margin-right':'auto',})),]
                                                            ),]),]),
                                    # richtige App
                                    dcc.Tab(label='Data Transformation',id='Data-trans',children=[]),
                                    dcc.Tab(label='Data_Exploration',id='Data-exp',children=[])
                                    
                            ])
                            ]),
                    dcc.Store(id='store',storage_type='session'),
                            ],fluid=True)
@app.callback(
    [Output('store','data'),
    Output('loading_info','children')],
    State('Path','value'),
    Input('Load-Data-button','n_clicks'), 
    State('change_dtypes','value'),prevent_initial_call=True)
def load_data(Path,n_clicks,change_dtypes):
    if Path is None:
        return[{},'Welcome to my Rapid Viz Tool! To start provide a Link to your Data']
    if n_clicks and ctx.triggered_id=='Load-Data-button':
        if Path:
            try:
                #df=px_data()
                Path=Path.strip('\"')
                df=read_data(os.path.normpath(Path))
            except:
                return [{},html.H6(children='The data was not loaded sucessfully! It seems the format you provided is not supported, the data is corrupt, or the path is not valid!',style={'color':f'{colors["Error"]}'})]    
            #check box
            if change_dtypes=='Automatically convert datatypes':
                df=df.convert_dtypes()
            return [df.to_dict('records'),html.H6(children='Data Loaded Sucessfully!',style={'color':f'{colors["Sucess"]}'})]
        else:
            return [{},html.H6(children='The data was not laoded sucessfully! You must specify a valid Path',style={'color':f'{colors["Error"]}'})]
    


# callbacks for Data Transformation Layout
#TODO build trasnrom Columns on trans_table
@app.callback(Output('Data-trans','children'),
    Input('store','data'),prevent_initial_call=True)
def update_trans_layout(data):
    if ctx.triggered_id==('store'):
        if data:
            df=pd.DataFrame.from_records(data)                 
            return  [dbc.Row(create_table(df,id='trans_table',renameable=True)),
                 dbc.Row([dbc.Col([html.H4('Transform Columns'),dbc.Row([dcc.Dropdown(options=df.columns,id='trans-dropdown',placeholder='Select Column to transform',style=min_style)]),dbc.Row([dbc.Col(html.Button('Label Encode Column',id='label-encode-button',style=min_style)),dbc.Col(html.Button('Scale Column Min/Max',id='scale-min/max-button',style=min_style)),dbc.Col(html.Button('Standardize Column',id='standardize-button',style=min_style))]),dbc.Row([dcc.Dropdown(options=['object','int64','float64','datetime64[ns]','bool'],id='dtypes-dropdown',placeholder='Select Column to transform',style=min_style),html.Button('Change Data Type of the selected column',id='change-dtype-button',style=min_style),html.Div(id='dtype-div',style=min_style)])]),
                          dbc.Col([dbc.Row(dcc.Checklist(['Scale all columns Min/Max','Standardize all columns'],[],id='scale-checklist'),style=min_style),dbc.Row(html.Button('Confirm Transformation',id='confirm-trans-button'),style=min_style)])])]
               
@app.callback(
        Output('trans_table','data'),
        Output('dtype-div','children'),
        State('trans_table','data'),
        State('trans-dropdown','value'),
        Input('label-encode-button','n_clicks'),
        Input('standardize-button','n_clicks'),
        Input('scale-min/max-button','n_clicks'),
        State('scale-checklist','value'),
        Input('confirm-trans-button','n_clicks'),
        Input('change-dtype-button','n_clicks'),
        State('dtypes-dropdown','value')
        ,prevent_initial_call=True)
def transform_data(data,column,label,standard,scale,checklist,confirm,change_dtypes_button,dtype):
    df=pd.DataFrame.from_records(data)
    if column:
        if ctx.triggered_id=='change-dtype-button':
            try:
                df[column]=df[column].astype(dtype)
                return df.to_dict("records"),html.H6(children=f'Changing the data type to {dtype} was scessfully!',style={'color':f'{colors["Sucess"]}'})
            except: return df.to_dict("records"), html.H6(children=f'Changing the data type to {dtype} was NOT scessfully! It seems the conversation to {dtype} for the column {column} is not possible',style={'color':f'{colors["Error"]}'})
        if ctx.triggered_id=='label-encode-button':
            df[column]=preprocessing.LabelEncoder().fit_transform(df[column])
            return df.to_dict("records"),html.H6(children=f'The Column {column} was Label Encoded sucessfully!',style={'color':f'{colors["Sucess"]}'})
        if ctx.triggered_id=='standardize-button':
            df[column]=preprocessing.StandardScaler().fit_transform(df[column].values.reshape(-1, 1))
            return df.to_dict("records"),html.H6(children=f'The Column {column} was Standardized sucessfully!',style={'color':f'{colors["Sucess"]}'})
        if ctx.triggered_id=='scale-min/max-button':
            df[column]=preprocessing.MinMaxScaler().fit_transform(df[column].values.reshape(-1, 1))
            return df.to_dict("records"),html.H6(children=f'The Column {column} was Scaled sucessfully!',style={'color':f'{colors["Sucess"]}'})
    else: raise PreventUpdate
    if ctx.triggered_id=='confirm-trans-button':
        if "Scale all columns Min/Max" in checklist: 
            dff=pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(df),columns=df.columns)
            return dff.to_dict("records"),html.H6(children=f'The Data was transformed sucessfully! You can now proceed to the Data Exploration Tab',style={'color':f'{colors["Sucess"]}'})
        if "Standardize all columns" in checklist: 
            dff=pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(df),columns=df.columns)
            return dff.to_dict("records"),html.H6(children=f'The Data was transformed sucessfully! You can now proceed to the Data Exploration Tab',style={'color':f'{colors["Sucess"]}'})
        else: return df.to_dict("records"),html.H6(children=f'The Data was transformed sucessfully! You can now proceed to the Data Exploration Tab',style={'color':f'{colors["Sucess"]}'})

@app.callback(Output('Data-exp','children'),
    State('trans_table','data'),
    Input('confirm-trans-button','n_clicks'))
def update_table(data,confirm):
    if data:
        df=pd.DataFrame.from_records(data)
        return dbc.Row(create_table(df,id='data_table',renameable=False)),dbc.Row(dcc.Tabs(id='graphs',children=[create_Tab1(df),create_Tab2(df),create_Tab3(df),create_Tab4(df),create_Tab5(df),create_Tab6(df),create_Tab8(df),create_Export()])),
    
#--------------------------Graph---------callbacks-------------
@app.callback(Output('stats-table','data'),
              State('data_table','data'),
              Input('data_table','derived_virtual_data'),
            Input('data_table','derived_virtual_selected_rows'))
def update_stats(data,rows,derived_virtual_selected_rows):
    df=pd.DataFrame.from_records(data)
    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows=[]
    dff=df if rows is None else pd.DataFrame(rows)
    dfff=dff.describe(include='all')
    dfff.insert(0,'statistical values',dfff.index)
    return dfff.to_dict('records')

@app.callback(
        Output('stat-export','children'),
        Input('export-stats','n_clicks'),
        State('stats-name','value'),
        State('stats-table','data'),
        State('Save_Path','value'),
)
def export_Stats(n_clicks,name,data,save_path):
    if ctx.triggered_id=='export-stats':
        df=pd.DataFrame.from_records(data)
        if not name:
            name='stats'
        if save_path:
            path=os.path.join(save_path,f'{name}.csv')
        else:
            path=f'{name}.csv'
        df.to_csv(path)
        return html.H5(f"Statistics are saved sucessfully under '{path}'",style={'color':f'{colors["Sucess"]}'})


@app.callback(
    Output('PC-Graph','figure'),
    State('data_table','data'),
    Input('data_table','derived_virtual_data'),
    Input('data_table','derived_virtual_selected_rows'),
    Input('PC-color-dropdown','value'),
    Input('PC-Lower-Bound','value'),
    Input('PC-Upper-Bound','value'),
    Input('PC-name','value'),
    Input('Save_Path','value'),
    Input('PC-save-plot','n_clicks'),prevent_initial_call=True,
)
def update_PC_graph(data,rows,derived_virtual_selected_rows,color_column,up,low,title,save_path,save):
    df=pd.DataFrame.from_records(data)
    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows=[]
    dff=df if rows is None else pd.DataFrame(rows)
    #TODO upper value and lower value are not in use right now
    fig=px.parallel_coordinates(dff,color=color_column,template=figure_template,color_continuous_scale=color_scale)
    if title:
        fig.update_layout(title=title)
    if ctx.triggered_id=='PC-save-plot':
        save_plot(fig,name=f'{title}.html',save_path=save_path)
    return fig

@app.callback(
    Output('Col-Graph','figure'),
    State('data_table','data'),
    Input('data_table','derived_virtual_data'),
    Input('data_table','derived_virtual_selected_rows'),
    Input('Col-color-dropdown','value'),
    Input('Col-x-dropdown','value'),
    Input('Col-pattern-dropdown','value'),
    Input('Col-name','value'),
    Input('Save_Path','value'),
    Input('Col-save-plot','n_clicks'),prevent_initial_call=True,
)
def update_Col_graph(data,rows,derived_virtual_selected_rows,color_column,x,pattern,title,save_path,save):
    if x:
        df=pd.DataFrame.from_records(data)
        if derived_virtual_selected_rows is None:
            derived_virtual_selected_rows=[]
        dff=df if rows is None else pd.DataFrame(rows)
        fig=px.histogram(dff,x=x,color=color_column,marginal='box',pattern_shape=pattern,template=figure_template,color_discrete_sequence=color_scale)
        if title:
            fig.update_layout(title=title)
        if ctx.triggered_id=='Col-save-plot':
            save_plot(fig,name=f'{title}.html',save_path=save_path)
        return fig
    else: return {}

@app.callback(
    Output('SC-Graph','figure'),
    State('data_table','data'),
    Input('data_table','derived_virtual_data'),
    Input('data_table','derived_virtual_selected_rows'),
    Input('SC-color-dropdown','value'),
    Input('SC-x-dropdown','value'),
    Input('SC-y-dropdown','value'),
    Input('SC-size-dropdown','value'),
    Input('SC-name','value'),
    Input('Save_Path','value'),
    Input('SC-save-plot','n_clicks'),prevent_initial_call=True,
)
def update_SC_graph(data,rows,derived_virtual_selected_rows,color_column,x,y,size,title,save_path,save):
    if x or y:
        df=pd.DataFrame.from_records(data)
        if derived_virtual_selected_rows is None:
            derived_virtual_selected_rows=[]
        dff=df if rows is None else pd.DataFrame(rows)
        fig=px.scatter(dff,x=x,y=y,color=color_column,trendline='ols',size=size,marginal_x='box',marginal_y='box',template=figure_template,color_continuous_scale=color_scale)
        if title:
            fig.update_layout(title=title)
        if ctx.triggered_id=='SC-save-plot':
            save_plot(fig,name=f'{title}.html',save_path=save_path)
        return fig
    else: raise PreventUpdate

@app.callback(
    Output('SC3D-Graph','figure'),
    State('data_table','data'),
    Input('data_table','derived_virtual_data'),
    Input('data_table','derived_virtual_selected_rows'),
    Input('SC3D-color-dropdown','value'),
    Input('SC3D-x-dropdown','value'),
    Input('SC3D-y-dropdown','value'),
    Input('SC3D-z-dropdown','value'),
    Input('SC3D-size-dropdown','value'),
    Input('SC3D-name','value'),
    Input('Save_Path','value'),
    Input('SC3D-save-plot','n_clicks'),prevent_initial_call=True,
)
def update_SC3D_graph(data,rows,derived_virtual_selected_rows,color_column,x,y,z,size,title,save_path,save):
    if x or y or z:
        df=pd.DataFrame.from_records(data)
        if derived_virtual_selected_rows is None:
            derived_virtual_selected_rows=[]
        dff=df if rows is None else pd.DataFrame(rows)
        fig=px.scatter_3d(dff,x=x,y=y,z=z,color=color_column,size=size,template=figure_template,color_continuous_scale=color_scale)
        if title:
            fig.update_layout(title=title)
        if ctx.triggered_id=='SC3D-save-plot':
            save_plot(fig,name=f'{title}.html',save_path=save_path)
        return fig
    else: raise PreventUpdate
        
@app.callback(
    Output('Ridge-Graph','figure'),
    State('data_table','data'),
    Input('data_table','derived_virtual_data'),
    Input('data_table','derived_virtual_selected_rows'),
    Input('Ridge-color-dropdown','value'),
    Input('Ridge-x-dropdown','value'),
    Input('Ridge-y-dropdown','value'),
    Input('Ridge-name','value'),
    Input('Save_Path','value'),
    Input('Ridge-save-plot','n_clicks'),prevent_initial_call=True,
)
def update_Ridge_graph(data,rows,derived_virtual_selected_rows,color_column,x,y,title,save_path,save):
    if y or x:
        df=pd.DataFrame.from_records(data)
        if derived_virtual_selected_rows is None:
            derived_virtual_selected_rows=[]
        dff=df if rows is None else pd.DataFrame(rows)
        #maybe use https://github.com/tpvasconcelos/ridgeplot
        fig=px.violin(dff,x=x,y=y,orientation='h',color=color_column,template=figure_template,color_discrete_sequence=color_scale).update_traces(side='positive',width=8)
        if title:
            fig.update_layout(title=title)
        if ctx.triggered_id=='Ridge-save-plot':
            save_plot(fig,name=f'{title}.html',save_path=save_path)
        return fig   
    else: raise PreventUpdate

@app.callback(
    Output('Corr-Graph','figure'),
    State('data_table','data'),
    Input('data_table','derived_virtual_data'),
    Input('data_table','derived_virtual_selected_rows'),
    Input('Corr-type-dropdown','value'),
    Input('Corr-name','value'),
    Input('Save_Path','value'),
    Input('Corr-save-plot','n_clicks'),prevent_initial_call=True,
)
def update_Corr_graph(data,rows,derived_virtual_selected_rows,corr_type,title,save_path,save):
    df=pd.DataFrame.from_records(data)
    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows=[]
    dff=df if rows is None else pd.DataFrame(rows)
    if corr_type:
        cor=dff.corr(corr_type)
        fig=px.imshow(cor,text_auto=True,template=figure_template,color_continuous_scale=color_scale)
        if title:
            fig.update_layout(title=title)
        if ctx.triggered_id=='Corr-save-plot':
            save_plot(fig,name=f'{title}.html',save_path=save_path)
        return fig            
    else: raise PreventUpdate

@app.callback(
        Output('Export-div','children'),
    Input('Export Data','n_clicks'),
    State('Export-name','value'),
    Input('data_table','derived_virtual_data'),
    Input('data_table','derived_virtual_selected_rows'),
    State('data_table','data'),
    State('Save_Path','value'),
)
def export_data(export,name,rows,derived_virtual_selected_rows,data,save_path):
    if ctx.triggered_id=='Export Data':
        if name:
            df=pd.DataFrame.from_records(data)
            if derived_virtual_selected_rows is None:
                derived_virtual_selected_rows=[]
            dff=df if rows is None else pd.DataFrame(rows)
            filename,ext=os.path.splitext(name)
            if save_path:
                path=os.path.join(save_path,name)
            else:
                path=name
            if ext:
                try:
                    if ext=='.csv':
                        dff.to_csv(path)
                    elif ext=='.parquet':
                        dff.to_parquet(path)
                    elif ext=='.xlsx':
                        dff.to_excel(path)
                    return html.H6(children=f'The Data was exportet to {ext} sucessfully! You can find the export under "{path}"',style={'color':f'{colors["Sucess"]}'})
                except:
                    return html.H6(children=f'The Export failed! Please provide a file name with the targeted extension. (Supportet are: *.xlsx,*.parquet,*.csv) ',style={'color':f'{colors["Error"]}'})
            else: html.H6(children=f'Please provide a file name with the targeted extension. (Supportet are: *.xlsx,*.parquet,*.csv)',style={'color':f'{colors["Error"]}'})
        else:
            return html.H6(children=f'Please provide a file name with the targeted extension. (Supportet are: *.xlsx,*.parquet,*.csv)',style={'color':f'{colors["Error"]}'})

if __name__ == "__main__":
    app.run(debug=True)
