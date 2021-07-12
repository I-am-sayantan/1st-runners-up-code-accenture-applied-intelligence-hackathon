import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 

#from PIL import Image

#app=Flask(__name__)
#Swagger(app)

pickle_in = open("finalized_model.pkl","rb")
classifier=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
       
def predict_note_authentication(Breathing_Problem,Fever,Dry_Cough,Sore_throat,Hyper_Tension,Abroad_travel,Contact,Attended,Visited,Family):
    
   
    X=np.array([[Breathing_Problem,Fever,Dry_Cough,Sore_throat,Hyper_Tension,Abroad_travel,Contact,Attended,Visited,Family]])
    X[X=='yes'] = 1
    X[X=='no'] = 0
    prediction=classifier.predict(X)
    if prediction==1:
      prediction="Our AI model feels after analyzing your symptoms , that you can be a potential covid patient , please contact your nearest medical care centre. "
    else:
      prediction="Our AI model feels after analyzing your symptoms, that you are safe, so be safe and wear mask."
    print(prediction)
    return prediction



def main():


    html_temp = """
            <div style="padding:10px;">
            <center>
           <img src='https://user-images.githubusercontent.com/50532530/125352577-10f86c80-e37f-11eb-8a69-71de3a5a1aa3.jpg' width='512' height='300'  style="align:center;">
           </center>
            </div>
            <br>
            """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.title("RECOMMENDING RESTRICTIONS FOR GIVEN COUNTRY")

    @st.cache(allow_output_mutation=True)

    def restrictions(i):
        data= df[df["CountryName"]==i].reset_index()
        data.drop(['index','CountryName'],axis=1,inplace=True)
        data["newcases"]= data["ConfirmedCases"]-data["ConfirmedCases"].shift( periods=1)
        data.at[0,'newcases']=data["ConfirmedCases"].values[0]
        data.set_index('Date',inplace=True)
        data = data[columns]
        col = columns[1:]
        last_day = data.index[-1]
        fcol = []
        for j in col:
            if data.loc[last_day,j] == 0:
                data.drop(j,axis=1)
            else:
                fcol.append(j[14:-4])
        fcol = set(fcol)
        return fcol



    def load_model():
        data1=pd.read_csv("Clusters.csv")
        df=pd.read_csv("COVID_gov_complete_29_03.csv",index_col=0)
        return data1,df

    with st.spinner('Loading Model Into Memory....'):
    
        data1,df = load_model()
        clusterwise_country=['Greece','Canada','Barbados','Angola','Mexico','Zimbabwe','Iceland']
        columns=['newcases', 
            'Days_since_S1_School closing_1.0', 
            'Days_since_S1_School closing_2.0',
            'Days_since_S2_Workplace closing_1.0',
            'Days_since_S2_Workplace closing_2.0',
            'Days_since_S3_Cancel public events_1.0',
            'Days_since_S3_Cancel public events_2.0',
            'Days_since_S4_Close public transport_1.0',
            'Days_since_S4_Close public transport_2.0',
            'Days_since_S5_Public information campaigns_1.0',
            'Days_since_S6_Restrictions on internal movement_1.0',
            'Days_since_S6_Restrictions on internal movement_2.0',
            'Days_since_S7_International travel controls_1.0',
            'Days_since_S7_International travel controls_2.0',
            'Days_since_S7_International travel controls_3.0',]

        texts = st.text_input('Enter country name ...')

    if texts:
        try:
            texts=texts.lower()
            texts=texts[0].upper()+texts[1:]
            st.write("Response :")
            with st.spinner('Searching for answers.....'):
                n=np.array(data1[data1["Country/Region"]==texts]["clusters"])[0]
                final=restrictions(clusterwise_country[n])
                html_temp = """
            <div style="background-color:tomato;padding:10px">
            <h2 style="color:white;text-align:center;">These are the restrictions suggested to control increase in the number of cases: </h2>
            </div>
            <br>
            """
                st.markdown(html_temp,unsafe_allow_html=True)
                #st.write(html_temp)
                for i in final:
                    st.write(i) 
                    st.write("-------------------------------------------")
        except:
            st.write("Enter a valid country name")
            st.write("-------------------------------------------")


    html_temp = """
            <div style="padding:10px;">
            <center>
           <a><img src='https://user-images.githubusercontent.com/50532530/125205463-01eebd00-e2a0-11eb-86e8-c6cdedb8ecda.png' ></a>
           
          
           </center>
            </div>
            <br>
            """
    st.markdown(html_temp,unsafe_allow_html=True)

    st.title("POPULATIONS THAT HAVE THE ​HIGHEST RISK OF ​CONTRACTING COVID-19")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Type "yes" if you it is true otherwise type "no" </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Breathing_Problem = st.text_input("Breathing Problem","Type Here")
    Fever = st.text_input("Fever","Type Here")
    Dry_Cough = st.text_input("Dry Cough","Type Here")
    Sore_throat = st.text_input("Sore throat","Type Here")
    Hyper_Tension = st.text_input("Hyper Tension","Type Here")
    Abroad_travel = st.text_input("Abroad travel","Type Here")
    Contact = st.text_input("Contact with COVID Patient","Type Here")
    Attended = st.text_input("Attended Large Gathering","Type Here")
    Visited = st.text_input("Visited Public Exposed Places","Type Here")
    Family = st.text_input("Family working in Public Exposed Places","Type Here")
    
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(Breathing_Problem,Fever,Dry_Cough,Sore_throat,Hyper_Tension,Abroad_travel,Contact,Attended,Visited,Family)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("More solution ")
        st.text("Comming soon")

    

if __name__=='__main__':
    main()