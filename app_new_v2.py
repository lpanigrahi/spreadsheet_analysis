
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import json
import os
from langchain.tools import Tool
# from langchain.utilities import GoogleSearchAPIWrapper
import uuid
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
import re
import spacy
from keywords import keywords
nlp = spacy.load('en_core_web_lg')

import matplotlib.pyplot as plt
from langchain.agents.agent_types import AgentType
os.environ["OPENAI_API_KEY"] = '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'

import shutil
import sys
import tempfile

from contextlib import contextmanager

def get_spreadsheet_summary(df, query):
    agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    max_execution_time=200,
    handle_parsing_errors= True
    )
    return agent.run(query)

summary_prompt = """ using python tool, do a detailed analysis of this dataframe. The analysis should include the following aspects:\

         1. Descriptive Statistics: \
            Summary statistics for each column (mean, median, mode, standard deviation, etc.). \
            Distribution of values in each column (e.g., histograms, box plots). \
         2. Data Quality: \
            Identification of missing values, if any. \
            Detection of outliers and anomalies. \
            Assessment of data consistency and integrity. \
         3. Correlation Analysis: \
            Correlation matrix showing relationships between numerical columns. \
            Identification of any significant correlations and potential multicollinearity issues.\
         4. Categorical Data Analysis: \
            Frequency distribution of categorical variables.\
            Analysis of categorical data relationships and associations.\
         5. Time Series Analysis (if applicable): \
            Trend analysis, seasonality, and cyclic patterns for time-based data.\
         7. Key Insights and Recommendations: \
            Highlight any significant findings or patterns. \
            Provide actionable recommendations based on the analysis. \
        Please ensure the analysis is thorough and presented in a clear and tabular format and don't generate any graphs \
"""


def main():
    # load_dotenv()
    # search = GoogleSearchAPIWrapper()


    st.set_page_config(page_title="Scolo", page_icon=":moneybag:")
    st.markdown("""
        <style>
        .subtitle {
            font-size: 20px;
            color: #555555;
        }
        </style>
        """, unsafe_allow_html=True)
    st.header("🦜🔗 AI Spreadsheet Analyst")
    st.markdown('<p class="subtitle">Upload a Excel file with headers as the first line for optimal results. Rest assured, your documents are read locally on your device, ensuring strict privacy compliance.</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an Excel file", type='xlsx')

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



    if uploaded_file is not None:
        with tempdir() as base_dir:
            unique_filename = str(uuid.uuid4()) + '.xlsx'
            uploads_folder = base_dir
            os.makedirs(uploads_folder, exist_ok=True)  # Create 'uploads' folder if it doesn't exist
            file_path = os.path.join(uploads_folder, unique_filename)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getvalue())

            df = pd.read_excel(uploaded_file)
            agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

        if st.checkbox("Show top five records from spreadsheet"):
            with st.expander('Data'):
                st.table(df.head(5).style.set_properties(**{'text-align': 'center'}).set_table_styles([{
                    'selector': 'th',
                    'props': [('text-align', 'center')]
                }]))

        # account_owner = st.text_input("Enter subject's name to help the AI understand the context of the data (e.g. 'John Doe') ")

        # if not get_table_columns(agent):
        #     st.error("Unable to extract table columns. Please ensure that the file has columns for `date`, `debit`, `credit`, and `description`.")
        #     st.stop()
        # if account_owner:
        # if st.checkbox("👨‍💻 Chat with your CSV"):
        #     query = st.text_area("Insert your query")
        #     if st.button("Submit Query", type="primary"):
        #         table_columns = get_spreadsheet_summary(df, query)
        #         st.info(table_columns)
        #         st.success("Done!")

        if st.checkbox(" 📃  Generate Analysis"):
            with st.spinner(text="Analysing document..."):
                # table_columns = get_table_columns(agent)
                # top_ten_transactions = get_top_ten_transactions(df, table_columns)
                with st.expander('AI Suggested reports'):
                    with st.spinner(text="Genertaing reports..."):  
                        summary = get_spreadsheet_summary(df, summary_prompt)
                        st.info(summary)
                        st.success("Done!")
        
        if st.checkbox("👨‍💻 Chat with your CSV"):
            query = st.text_area("Insert your query")
            if st.button("Submit Query", type="primary"):
                query = "using python_repl_ast, from the dataframe  " + query 
                result = get_spreadsheet_summary(df, query)
                st.info(result)
                st.success("Done!")

    

if __name__ == "__main__":
    main()
