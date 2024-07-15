#Tag Extractor
import os
import uuid
import shutil
import tempfile
import pathlib
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from pydantic import BaseModel, Field
from typing import List
from contextlib import contextmanager
from IPython.display import display_html
from itertools import chain,cycle

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

from langchain import OpenAI

# Define the model to use

openai_api_key=OPENAI_API_KEY

model = ChatOpenAI(temperature=0,openai_api_key=OPENAI_API_KEY)



class Tagging_Inv(BaseModel):
    """Tag the piece of text with particular info."""
    key_01: str = Field(description="Name of the Exporter and Manufacturer, should be the content of 'Exporter & Manufacturer' field`")
    key_02: str = Field(description="Ship To (Consignee) and should be the content of 'Ship To (Consignee)' field")
    key_03: str = Field(description="Invoice to (Buyer if other than Consignee) and should be the content of 'Invoice to (Buyer if other than Consignee)' field")
    key_04: str = Field(description="Value of Port of Loading and should be the content of 'Port of Loading' field")
    key_05: str = Field(description="Country of Final Desitination and should be the content of 'Country of Final Desitination' field")
    key_06: str = Field(description="Final Delivery and should be the content of 'Final Delivery' field")
    key_07: str = Field(description="Port of Discharge and should be the content of 'Port of Discharge' field")
    key_08: str = Field(description="Final Destination and should be the content of 'Final Destination' field")
    key_09: str = Field(description="Payment and should be the content of 'Payment' field")
    key_10: str = Field(description="Terms of Delivery and should be the content of 'Terms of Delivery' field")
    key_11: str = Field(description="Invoice No and should be the content of 'Invoice No' field and extracteted information should be in numeric format")
    key_12: str = Field(description="Invoice Date and should be the content of 'Invoice Date' field and extracted output in timestamp format like 'dd/mm/yyyy'") 
    key_13: str = Field(description="Mentioned in Marks & Nos.and should be the content of 'Marks & Nos/ Container No.' field")
    key_14: str = Field(description="Description of Goods and should be the content of 'Description of Goods' field")
    key_15: str = Field(description="Value should be the content of 'Quantity' field and extract only the digits and output as numeric datatype")
    key_16: str = Field(description="Specified in Invoice along with Qty. shipped. and should be the content of 'Specified in Invoice along with Qty. shipped.' field")
    key_17: str = Field(description="should be the content of Amount Value field and currency in USD and the final output as numeric datatype with only digits")
    key_18: str = Field(description="should be the content of FOB Value(USD) field and currency in USD and the final output as numeric datatype with only digits")
    key_19: str = Field(description="should be the content of FOB in INR field and currency in INR and the final output as numeric datatype with only digits")
    key_20: str = Field(description="should be the content of 'Total Value(FC)' field and the final output as numeric datatype with only digits")
    key_21: str = Field(description="should be the content of 'Insurance Name' field and extract only the digits and output as numeric datatype")
    key_22: str = Field(description="should be the content of 'Freight' field and extract only the digits and output as numeric datatype")
    key_23: str = Field(description="Rate per Unit and should be the content of 'Rate per Unit' field")
    key_24: str = Field(description="'IGST @   __ %' and should be the content of 'IGST @   __ %' field")
    key_25: str = Field(description="Marks & Nos and should be the content of 'Marks & Nos/ Container No.' field")
    key_26: str = Field(description="Advance Licence No. & Date and should be the value of 'Advance Licence No. & Date' field")
    key_27: str = Field(description="Advance Licence Qty and should be the value of 'Advance Licence Qty' field")
    key_28: str = Field(description="should be the content of 'Gross Weight' field and the final output as numeric datatype with only digits and numbers")
    key_29: str = Field(description="should be the content of 'Net Weight' field and the final output as numeric datatype with only digits and numbers")
    key_30: str = Field(description="should be the content of 'Pallets' field and the final output as numeric datatype with only digits and numbers")
    key_31: str = Field(description="NA  and should be the content NA Value")
    key_32: str = Field(description="NA  and should be the content NA Value")

class Tagging_Cha(BaseModel):
    """Tag the piece of text with particular info."""
    key_01: str = Field(description="Name of the exporter and should be the content of 'Exporter's Name' field")
    key_02: str = Field(description="Consignee Name and should be the content of 'Consignee's Name' field")
    key_03: str = Field(description="Buyer Name and should be the content of 'Buyer's Name' field")
    key_04: str = Field(description="Port of Loading and should be the content of 'Port of Loading' field")
    key_05: str = Field(description="Final Desitination Country and should be the content of 'Final Desitination Country' field")
    key_06: str = Field(description="Final Desitination Port and should be the content of 'Final Desitination Port' field")
    key_07: str = Field(description="Port of Discharge and should be the content of 'Port of Discharge' field")
    key_08: str = Field(description="Country of Discharge and should be the content of 'Country of Discharge' field")
    key_09: str = Field(description="Nature of Payment and should be the content of 'Nature of Payment' field")
    key_10: str = Field(description="Nature of Contract and should be the value of 'Nature of Contract' field")
    key_11: str = Field(description="Invoice No and should be the content of 'Invoice No' field")
    key_12: str = Field(description="Invoice Date and should be the content of 'Invoice Date' field")
    # key_13: str = Field(description="RITC Code and should be mapping to the content of 'RITC Code' and extract final output by removing '#Pkg' value ")
    key_13: str = Field(description="content of 'RITC Code' field contain RITC Code and Pkg information , remove first single digit from 'RITC Code'")
    key_14: str = Field(description="Item Description and should be the content of 'Item Description' field")
    key_15: str = Field(description="Quantity and should be the content of 'Quantity' field")
    key_16: str = Field(description="Unit of Measurement and should be the content of 'Unit of Measurement' field")
    key_17: str = Field(description="Invoice Value(USD) and should be the content of 'Invoice Value (USD)' field and currency in USD")
    key_18: str = Field(description="FOB Value(USD) and should be the content of 'FOB Value(USD)' field and currency in USD")
    key_19: str = Field(description="FOB Value(Rs.) and should be the content of 'FOB Value(Rs.)' field and currency in Rs.")
    key_20: str = Field(description="should be the content of 'Total Value(FC)' field and the final output as numeric datatype")
    key_21: str = Field(description="should be the content of 'Insurance Name' field and extract only the digits and output as numeric datatype")
    key_22: str = Field(description="should be the content of 'Freight' field and extract only the digits and output as numeric datatype")
    key_23: str = Field(description="Rate per Unit and should be the content of 'Rate per Unit' field")
    key_24: str = Field(description="IGST Amount and should be the content of 'IGST Amount' field")
    key_25: str = Field(description="Marks & Nos and should be the content of 'Marks & Nos' field")
    key_26: str = Field(description="Advance Licence No. & Date and should be the value of 'Advance Licence No. & Date' field")
    key_27: str = Field(description="Advance Licence Qty and should be the value of 'Advance Licence Qty' field")
    key_28: str = Field(description="should be the content of 'Gross Weight' field and the final output as numeric datatype with only digits and numbers")
    key_29: str = Field(description="should be the content of 'Net Weight' field and the final output as numeric datatype with only digits and numbers")
    key_30: str = Field(description="should be the content of 'No of Packages' field and the final output as numeric datatype with only digits and numbers")
    key_31: str = Field(description="RODTEP Claim Rate and should be the content of 'RODTEP Claim Rate' field")
    key_32: str = Field(description="RODTEP Amount and should be the content of 'RODTEP Amount' field")


inv_map_dict = dict()
inv_map_dict.update({'key_01':"Exporter & Manufacturer", 'key_02':"Ship To (Consignee)" , 'key_03':"Invoice to (Buyer if other than Consignee)",
                    'key_04':"Port of Loading", 'key_05':"Country of Final Desitination", 'key_06':"Final Delivery",
                    'key_07':"Port of Discharge", 'key_08':"Final Destination", 'key_09':"Payment", 'key_10':"Terms of Delivery",
                    'key_11':"Invoice No", 'key_12':"Invoice Date", 'key_13':"Mentioned in Marks & Nos.", 'key_14':"Description of Goods",
                    'key_15':"Quantity", 'key_16':"Specified in Invoice along with Qty. shipped.", 'key_17':"Amount In USD", 
                    'key_18':"FOB Value in USD", 'key_19':"FOB in INR", 'key_20':"Total Value in (FC)", 'key_21':"Insurance",
                    'key_22':"Freight", 'key23':"Rate per Unit", 'key_24':"IGST @   __ %", 'key_25':"Marks & Nos", 'key_26':"Advance Licence No. & Date",
                    'key_27':"Advance Licence Qty", 'key_28':"Gross Weight", 'key_29':"Net Weight", 'key_30':"Pallets", 'key_31':"NA", 'key_32':"NA"})


cha_map_dict = dict()
cha_map_dict.update({'key_01':"Exporter's Name", 'key_02':"Consignee's Name" , 'key_03':"Buyer's Name",
                     'key_04':"Port of Loading", 'key_05':"Final Desitination Country", 'key_06':"Final Desitination Port",
                     'key_07':"Port of Discharge", 'key_08':"Country of Discharge", 'key_09':"Nature of Payment", 'key_10':"Nature of Contract",
                     'key_11':"Invoice No", 'key_12':"Invoice Date", 'key_13':"RITC Code", 'key_14':"Item Description",
                     'key_15':"Quantity", 'key_16':"Unit of Measurement", 'key_17':"Invoice Value (USD)", 
                     'key_18':"FOB Value (USD)", 'key_19':"FOB Value (Rs.)", 'key_20':"Total Value(FC)", 'key_21':"Insurance",
                     'key_22':"Freight", 'key23':"Rate per Unit", 'key_24':"IGST Amount", 'key_25':"Marks & Nos", 'key_26':"Advance Licence No. & Date",
                     'key_27':"Advance Licence Qty", 'key_28':"Gross Weight", 'key_29':"Net Weight", 'key_30':"No of Packages", 
                      'key_31':"RODTEP Claim Rate", 'key_32':"RODTEP Amount"})



inv_list = ['Exporter & Manufacturer','Ship To (Consignee)','Invoice to (Buyer if other than Consignee)','Port of Loading',
 'Country of Final Desitination','Final Delivery','Port of Discharge','Final Destination','Payment','Terms of Delivery',
 'Invoice No','Invoice Date','Mentioned in Marks & Nos.','Description of Goods','Quantity','Specified in Invoice along with Qty. shipped.',
 'Amount In USD','FOB Value in USD','FOB in INR','Total Value in (FC)','Insurance','Freight','Rate per Unit','IGST @   __ %','Marks & Nos',
 'Advance Licence No. & Date','Advance Licence Qty','Gross Weight','Net Weight','Pallets','NA','NA']

cha_list = ["Exporter's Name","Consignee's Name","Buyer's Name","Port of Loading","Final Desitination Country","Final Desitination Port",
 "Port of Discharge","Country of Discharge","Nature of Payment","Nature of Contrac","Invoice No","Invoice Date","RITC Code","Item Description",
 "Quantity","Unit of Measurement","Invoice Value (USD)","FOB Value (USD)","FOB Value (Rs.)","Total Value(FC)", "Insurance","Freight","Rate per Unit",
 "IGST Amount", "Marks & Nos","Advance Licence No. & Date","Advance Licence Qty","Gross Weight","Net Weight","No of Packages","RODTEP Claim Rate","RODTEP Amount"]

tagging_functions_Inv = [convert_pydantic_to_openai_function(Tagging_Inv)]
tagging_functions_Cha = [convert_pydantic_to_openai_function(Tagging_Cha)]

prompt = ChatPromptTemplate.from_messages([
    ("system", "Think carefully, and then tag the text as instructed"),
    ("user", "{input}")
])

model_with_functions_Inv = model.bind(
    functions=tagging_functions_Inv,
    function_call={"name": "Tagging_Inv"}
)

model_with_functions_Cha = model.bind(
    functions=tagging_functions_Cha,
    function_call={"name": "Tagging_Cha"}
)

tagging_chain_Inv = prompt | model_with_functions_Inv | JsonOutputFunctionsParser()
tagging_chain_Cha = prompt | model_with_functions_Cha | JsonOutputFunctionsParser()

@st.cache_data
def convert_df_to_csv(df):
  # IMPORTANT: Cache the conversion to prevent computation on every rerun
  return df.to_csv().encode('utf-8')

def highlight_greater(row):

    if row['Invoice Tag Values'].lower() != row['Checklist Tag Values'].lower():
        color = 'red'
    else:
        color = 'white'
    # elif row[A] == row[B]:
    #     color = 'green'

    background = ['background-color: {}'.format(color) for _ in row]

    return background


def main():

    st.set_option('deprecation.showPyplotGlobalUse', False)
    # st.set_page_config(page_title="Scolo", page_icon=":moneybag:")
    st.set_page_config(page_title="DigiHR", page_icon=":robot_face:", layout="centered")
    st.markdown("""
        <style>
        .subtitle {
            font-size: 20px;
            color: #555555;
        }
        </style>
        """, unsafe_allow_html=True)
    st.header("🦜🔗 Gen-AI Invoice & Checklist File Comparision")
    st.markdown('<p class="subtitle">Upload Invoice & Checklist file with .pdf as extensions. \
                For better result, file should be clearly visible.\
                For file uploading please follow the proper sequence.\
                Your documents are read locally on your device, ensuring strict privacy compliance.</p>', unsafe_allow_html=True)
    uploaded_file_inv = st.file_uploader("Choose Invoice file having extension  .pdf",type="pdf")
    uploaded_file_cha = st.file_uploader("Choose Checklist file having extension  .pdf", type="pdf")


    if "rerun_counter" not in st.session_state:
        st.session_state.rerun_counter = 0  


    @contextmanager
    def tempdir():
        path = tempfile.mkdtemp()
        try:
            yield path
        finally:
            try:
                shutil.rmtree(path)
            except IOError:
                sys.stderr.write('Failed to clean up temp dir {}'.format(path))



    if (uploaded_file_inv and uploaded_file_cha) is not None:
        with tempdir() as base_dir:
            try:
                file_extension = pathlib.Path(uploaded_file_inv.name).suffix
                unique_filename_inv = str(uuid.uuid4()) + file_extension
                unique_filename_cha = str(uuid.uuid4()) + file_extension
                uploads_folder = base_dir
                os.makedirs(uploads_folder, exist_ok=True)  # Create 'uploads' folder if it doesn't exist
                file_path_inv = os.path.join(uploads_folder, unique_filename_inv)
                file_path_cha = os.path.join(uploads_folder, unique_filename_cha)
                with open(file_path_inv, 'wb') as f1:
                    f1.write(uploaded_file_inv.getvalue())
                with open(file_path_cha, 'wb') as f2:
                    f2.write(uploaded_file_cha.getvalue())

                reader_inv = PdfReader(uploaded_file_inv)
                reader_cha = PdfReader(uploaded_file_cha)

                text_inv = [page.extract_text() for page in reader_inv.pages]
                text_cha = [page.extract_text() for page in reader_cha.pages]

            except:
                st.error("Error in uploaded file, pls try with different file..")
                st.stop()



        if st.checkbox(" 📃  Generate all tags from uploaded Invoice  & Checklist File "):
            with st.spinner(text="Analysing uploaded documents..."):
                with st.expander('Gen AI Extracted tags from Invoice & Checklist Files'):
                    with st.spinner(text="Genertaing All the Tags information..."):
                        try:  
                            inv_extract = tagging_chain_Inv.invoke({"input": text_inv})
                            cha_extract = tagging_chain_Cha.invoke({"input": text_cha})
                            df_inv = pd.DataFrame.from_dict(inv_extract,orient='index').transpose()
                            df_cha = pd.DataFrame.from_dict(cha_extract,orient='index').transpose()
                            df_temp=(pd.concat([df_inv.set_axis(inv_list, axis='columns').transpose().set_axis(['Extracted Invoice info..'], axis='columns').reset_index(),
                                                df_cha.set_axis(cha_list, axis='columns').transpose().set_axis(['Extracted Invoice info..'], axis='columns').reset_index()],
                                                axis=1, ignore_index=True)).set_axis(['Invoice Tag', 'Invoice Tag Values','Checklist Tag','Checklist Tag Values'], axis=1)
                            
                            st.dataframe(df_temp.style.apply(highlight_greater, axis=1))

                            # df_temp.style.set_properties(**{'text-align': 'center'}).set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
                            # st.markdown('<style>.col_heading{text-align: center;}</style>', unsafe_allow_html=True)
                            # df_temp.columns = ['<div class="col_heading">'+col+'</div>' for col in df_temp.columns] 
                            # st.write(df_temp.to_html(escape=False), unsafe_allow_html=True)
                        
                            # st.table(df_temp.set_axis(['Invoice Tag', 'Invoice Tag Values','Checklist Tag','Checklist Tag Values'], axis=1))
                            st.download_button(
                                label="Download data as CSV",
                                data=convert_df_to_csv(df_temp.set_axis(['Invoice Tag', 'Invoice Tag Values','Checklist Tag','Checklist Tag Values'], axis=1)),
                                file_name='tag_extract.csv',
                                mime='text/csv',
                                )

                            st.success("Done!")
                        except Exception as e:
                            st.error(e)
                            st.stop()

    

if __name__ == "__main__":
    main()

