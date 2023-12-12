import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import plotly.express as px
from streamlit_option_menu import option_menu
from sklearn.metrics.pairwise import cosine_similarity

comment='''Giving title to the page'''

st.title('Sales Forecast Dashboard')

comment='''Load the data'''

df1 = pd.read_csv('Global_Superstore2.csv',encoding='unicode_escape')

comment='''Create sidebar'''

st.sidebar.title("Filter data")

comment='''Sidebar dropdown mutiselection'''

region_list = st.sidebar.multiselect("Select Region", df1['Region'].unique(), default="Africa")
segment_list = st.sidebar.multiselect("Select Segment", df1['Segment'].unique(),default="Consumer")
sub_cat_list = st.sidebar.multiselect("Select Sub Category", df1['Sub-Category'].unique(),default="Chairs")

comment='''Sort and reset data by date and removing 00:00:00 from date'''

df1=df1.sort_values(by='Order Date')
df1=df1.reset_index(drop=True)
df1['Order Date'] = pd.to_datetime(df1['Order Date'], errors='coerce').dt.date

comment='''Crete start and end date for filter'''

start_date=df1['Order Date'].iloc[0]
end_date=df1['Order Date'].iloc[len(df1['Order Date'])-1]
start_date = st.sidebar.date_input('Start date', start_date)
end_date = st.sidebar.date_input('End date', end_date)
if start_date < end_date:
    st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
else:
    st.sidebar.error('Error: End date must fall after start date.')

comment='''Search for respctive input in the data'''

df_filtered=df1
if region_list!=[]:
    df_filtered = df_filtered[(df_filtered['Region'].isin(region_list))]

if segment_list!=[]:
    df_filtered = df_filtered[(df_filtered['Segment'].isin(segment_list))]

if sub_cat_list!=[]:
    df_filtered = df_filtered[(df_filtered['Sub-Category'].isin(sub_cat_list))]

if start_date!=[] and end_date!=[]:
    mask = (df_filtered['Order Date'] > start_date) & (df_filtered['Order Date'] <= end_date)
    df_filtered = df_filtered.loc[mask]

comment='''Create Filter checkbox'''

df6=pd.DataFrame()
is_check = st.sidebar.checkbox("Apply Filter")
df6=df_filtered
check=0
if is_check:
    check=1
    df6=df_filtered
    st.sidebar.subheader('Filter applied')

comment='''horizontal Menu'''

selected2 = option_menu(None, ["Revenue", "Product Bundle forecast", "Customer Segment", 'Country wise & Shipping'], 
menu_icon="cast", default_index=0, orientation="horizontal")

#####################################################
                    ##TAB 1##
#####################################################

##Functions##

comment='''Function for Product sales & Sub Category Barplot'''

def create_barplot_Sales(df11,rev_count11):
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
   
    if rev_count11.empty==0:
        st.bar_chart(rev_count11)
    if rev_count11.empty==1:
        st.subheader('No sales generated for selected filters')

comment='''Function for ploting overtime'''

def create_trend_Sales(df11,rev_count12):
    
    fig = plt.figure()
    
    if rev_count12.empty==0:
        st.line_chart(rev_count12, use_container_width=True)
    if rev_count12.empty==1:
        st.subheader('No revenue earned')

def create_piechart_Sales(df11,rev_count13,check):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    labels = df11['Market'].unique()

    list1_as_set = set(labels)
    list2 = rev_count13.index.tolist()
    intersection = list1_as_set.intersection(list2)

    if rev_count13.empty==0:
        fig = px.pie(rev_count13, values=rev_count13 ,names=rev_count13.index.tolist())
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=0),)
        st.plotly_chart(fig, use_container_width=True)
        
    if rev_count13.empty==1:
        st.subheader('No sales generated for selected filters')


#####################################################
                    ##TAB 2##
#####################################################

def createbundles(product_recs):
    pivot = pd.pivot_table(df2,index = 'Customer ID',columns = 'Product Name',values = 'Sub-Category',aggfunc = 'count')
    pivot.reset_index(inplace=True)
    pivot = pivot.fillna(0)
    pivot = pivot.drop('Customer ID', axis=1)

    #Building co-relation matrix based on bundles
    co_matrix = pivot.T.dot(pivot)
    np.fill_diagonal(co_matrix.values, 0)

    #calculating cosine similarity
    cos_score = pd.DataFrame(cosine_similarity(co_matrix))
    cos_score.index = co_matrix.index
    cos_score.columns = np.array(co_matrix.index)

    #Take top five scoring recs that aren't the original product
    product_recs = []
    for i in cos_score.index:
        product_recs.append(cos_score[cos_score.index!=i][i].sort_values(ascending = False)[0:2].index)
         
    product_recs = pd.DataFrame(product_recs)
    product_recs['recs_list'] = product_recs.values.tolist()
    product_recs.index = cos_score.index
    product_recs=product_recs.set_axis([ 'Recommendation 1', 'Recommendation 2','Recommendation List'], axis=1)
    st.table(product_recs)

#####################################################
                    ##TAB 3##
#####################################################

def create_piechart_cs(df3):
    st.header("Gross margin of products by category")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    margin_count1=(df3.groupby(['Sub-Category']).sum()['Gross margin']).reset_index()
    margin_count_value=margin_count1
    #Filter Negative Values
    margin_count1 = margin_count1[margin_count1['Gross margin'] > 0]

    if margin_count1.empty==0:
        fig = px.pie(margin_count1, values='Gross margin',labels=margin_count1['Sub-Category'].unique(),names=margin_count1['Sub-Category'].unique())
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=0),)
        st.plotly_chart(fig, use_container_width=True)
        
    if margin_count1.empty==1:
        st.subheader('No gross margin generated for the selected filters')
        st.table(margin_count_value)

def create_barplot_cs(df3):
    st.header("Segment wise profit generated")
    fig = plt.figure()
    rev_count31=df3.groupby(['Segment']).sum()['Profit']
    
    if rev_count31.empty==0:
        st.bar_chart(rev_count31)
    if rev_count31.empty==1:
        st.subheader('No profit generated for selected filters')

def create_somechart_cs(df3):
    rev_count32=(df3.groupby(['Customer Name','Segment']).sum('Sales')).reset_index()
   
    fig = px.scatter(
        rev_count32,
        x="Sales",
        y="Segment",
        size="Sales",
        color="Segment",
        hover_name="Customer Name",
        log_x=True,
        size_max=60,
    )

    st.header("Customer Segment wise sales")
    if rev_count32.empty==0:
       st.plotly_chart(fig, theme="streamlit")
    if rev_count32.empty==1:
        st.subheader('No sales generated for selected filters')

#####################################################
                    ##TAB 4##
#####################################################

comment=''' Function for barplot'''

def create_barplot_cw(df4,rev_count42):
    st.header("Preferred Ship Mode")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    rev_count2=df4.groupby(['Ship Mode']).count()['Order ID']
    # rev_count42
    if rev_count42.empty==0:
        st.bar_chart(rev_count2)
    if rev_count42.empty==1:
        st.subheader('No revenue earned')

comment=''' Function for barplot'''

def create_somechart_cw(df4):
    rev_count43=(df4.groupby(['Country']).sum('Sales'))
    rev_count43=(rev_count43['Sales'].sort_values(ascending=False).head(10)).reset_index()
   
    fig = px.histogram(
        rev_count43,
        x="Sales",
        y="Country",
        color="Country",
        hover_name="Country",
        log_x=True,
        marginal="rug"
    )

    st.header("Top 10 Country wise sales")
    if df4.empty==0:
       st.plotly_chart(fig, theme="streamlit")
    if df4.empty==1:
        st.subheader('No sales generated for selected filters')
        
def create_areachart_cw(df4):
    st.header("Top 20 City wise Profit")
    rev_count44=(df4.groupby(['City']).sum('Profit'))
    rev_count44=(rev_count44['Profit'].sort_values(ascending=False).head(20)).reset_index()
    
    fig = px.scatter(rev_count44, x="Profit", y="City", color="City", 
                   hover_data=rev_count44.columns)
    
    if df4.empty==0:
        st.write(fig)
        #st.area_chart(df4,x="City",y="Profit")   
    if df4.empty==1:
        st.subheader('No profit generated for selected filters')

#####################################################
        ##Graph Calls for Funtions##
#####################################################

if selected2 == "Revenue":

    #Applying data filters and groups
    df11=df_filtered
    rev_count11=df11.groupby(['Sub-Category']).sum()['Sales']
    rev_count12=df11.groupby(['Product Name']).sum()['Sales']
    rev_count12=rev_count12.sort_values(ascending=False).head(10)
    rev_count13=df11.groupby(['Market']).sum()['Sales']

    df11["Order Date"] = pd.to_datetime(df11["Order Date"], dayfirst = True)
    df11["Order Date"] = pd.DataFrame([df11["Order Date"]]).transpose()
  
    df11['month']= df11["Order Date"].dt.month
    rev_count14=df11.groupby(['month']).sum()['Sales']
    
    st.header("Sales by Sub Category")
    create_barplot_Sales(df11,rev_count11)
    
    st.header("Sales by Products")
    create_barplot_Sales(df11,rev_count12)

    st.header("Sales by Market")
    create_piechart_Sales(df11,rev_count13,check)

    st.header("Revenue over Time")
    create_trend_Sales(df11,rev_count14)

elif selected2 == "Product Bundle forecast":
    df2=df_filtered
    createbundles(df2)

elif selected2 == "Customer Segment":
    df3=df_filtered
    df3['Gross margin']=df3['Profit']/df3['Sales']*100
    
    create_piechart_cs(df3)
    create_barplot_cs(df3)
    create_somechart_cs(df3)

elif selected2 == "Country wise & Shipping":
    df4=df_filtered
    rev_count42=df4.groupby(['Ship Mode']).sum()['Sales']
    create_barplot_cw(df4,rev_count42)
    create_areachart_cw(df4)
    create_somechart_cw(df4)
