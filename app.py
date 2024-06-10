#Spreadsheet Analyzer
import re
import shutil
import sys
import tempfile
import pathlib
import json
import os
import uuid
import streamlit as st
import pandas as pd
from langchain.tools import Tool
from langchain.output_parsers import ResponseSchema
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

import matplotlib.pyplot as plt
from langchain.agents.agent_types import AgentType
from contextlib import contextmanager


# import openai
from openai import OpenAI



client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY"),
)

# Define the model to use

MODEL_NAME = "gpt-3.5-turbo"

def get_spreadsheet_summary(df, query):
    agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    max_execution_time=None,
    max_iterations=120,
    handle_parsing_errors= True
    )
    return agent.run(query)



def handle_openai_query(df, column_names):
    """
    Handle the OpenAI query and display the response.

    Parameters:
    - df: DataFrame containing the data
    - column_names: List of column names in the DataFrame
    """

    # Create a text area for user input
    query = st.text_area(
        "Enter your Prompt:",
        placeholder="Prompt tips: Use plotting related keywords such as 'Plots' or 'Charts' or 'Subplots'. Prompts must be concise and clear, example 'Bar plot for the first ten rows.'",
        help="""
            How an ideal prompt should look like? *Feel free to copy the format and adapt to your own dataset.*
            
            ```
            -Subplot 1: Plot stacked barplot for technician and request status
            -Subplot 2: Plot Bar plot of the counts of values in the priority column.
            -Subplot 3: Plot pie chart for priority column.
            -Subplot 4: Plot pie chart for territory column.
            ```
            """,
    )

    # If the "Get Answer" button is clicked
    if st.button("Generate Graph"):
        # Ensure the query is not empty
        if query and query.strip() != "":
            # Define the prompt content
            prompt_content = f"""
            The dataset is ALREADY loaded into a DataFrame named 'df'. DO NOT load the data again.
            
            The DataFrame has the following columns: {column_names}
            
            Before plotting, ensure the data is ready:
            1. Check if columns that are supposed to be numeric are recognized as such. If not, attempt to convert them.
            2. Handle NaN values by filling with mean or median.
            
            Use package Pandas and Matplotlib ONLY.
            Provide SINGLE CODE BLOCK with a solution using Pandas and Matplotlib plots in a single figure to address the following query:
            
            {query}

            - USE SINGLE CODE BLOCK with a solution. 
            - Do NOT EXPLAIN the code 
            - DO NOT COMMENT the code. 
            - ALWAYS WRAP UP THE CODE IN A SINGLE CODE BLOCK.
            - The code block must start and end with ```
            
            - Example code format ```code```
        
            - Colors to use for background and axes of the figure : #F0F0F6
            - Try to use the following color palette for coloring the plots : #8f63ee #ced5ce #a27bf6 #3d3b41
            
            """

            # Define the messages for the OpenAI model
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful Data Visualization assistant who gives a single block without explaining or commenting the code to plot. IF ANYTHING NOT ABOUT THE DATA, JUST politely respond that you don't know.",
                },
                {"role": "user", "content": prompt_content},
            ]

            # Call OpenAI and display the response
            with st.status("📟 *Prompting is the new programming*..."):
                with st.chat_message("assistant", avatar="📊"):
                    botmsg = st.empty()
                    response = []
                    for chunk in client.chat.completions.create(
                        model=MODEL_NAME, messages=messages, stream=True
                    ):
                        # text = chunk.choices[0].get("delta", {}).get("content")
                        # print ('Test print>>>>>>>', chunk.choices[0])
                        text = chunk.choices[0].delta.content
                        if text:
                            response.append(text)
                            result = "".join(response).strip()
                            botmsg.write(result)
            execute_openai_code(result, df, query)


def extract_code_from_markdown(md_text):
    """
    Extract Python code from markdown text.

    Parameters:
    - md_text: Markdown text containing the code

    Returns:
    - The extracted Python code
    """
    # Extract code between the delimiters
    code_blocks = re.findall(r"```(python)?(.*?)```", md_text, re.DOTALL)

    # Strip leading and trailing whitespace and join the code blocks
    code = "\n".join([block[1].strip() for block in code_blocks])

    return code

def execute_openai_code(response_text: str, df: pd.DataFrame, query):
    """
    Execute the code provided by OpenAI in the app.

    Parameters:
    - response_text: The response text from OpenAI
    - df: DataFrame containing the data
    - query: The user's query
    """

    # Extract code from the response text
    code = extract_code_from_markdown(response_text)

    # If there's code in the response, try to execute it
    if code:
        try:
            exec(code)
            st.pyplot()
        except Exception as e:
            error_message = str(e)
            st.error(
                f"📟 Apologies, failed to execute the code due to the error: {error_message}"
            )
            st.warning(
                """
                📟 Check the error message and the code executed above to investigate further.

                Pro tips:
                - Tweak your prompts to overcome the error 
                - Use the words 'Plot'/ 'Subplot'
                - Use simpler, concise words
                - Remember, I'm specialized in displaying charts not in conveying information about the dataset
            """
            )
    else:
        st.write(response_text)


summary_prompt = """"using python tool,do a detailed analysis of this dataframe, covering the following aspects:

            Descriptive Statistics:
                    Calculate and summarize the mean, median, mode, standard deviation, minimum, and maximum for each numerical column.
                    Provide histograms and box plots to show the distribution of values in each numerical column.

            Data Quality Assessment:
                    Identify and quantify missing values in each column.
                    Detect and describe any outliers or anomalies in the data.
                    Assess the overall data consistency and integrity.

            Correlation Analysis:
                    Generate a correlation matrix for numerical columns.
                    Highlight any significant correlations and discuss potential multicollinearity issues.
            
            Categorical Data Analysis:
                    Provide frequency distributions for each categorical column.
                    Analyze the relationships and associations between different categorical variables.
                    Perform chi-square tests for independence where applicable.
            
            Segment Analysis:
                    Analyze different customer segments and their purchasing behaviors.
                    Provide insights into how different segments interact with various products and discounts.

            Time Series Analysis (if applicable):
                    Analyze trends, seasonality, and cyclic patterns in the data if there are time-based columns.
                    Provide line plots and other relevant visualizations to illustrate these patterns.

            Revenue and Sales Analysis(if applicable otherwise ignore):
                    Examine the relationship between discounts and sales/revenue.
                    Identify the top-performing products and regions.
                    Analyze the impact of different customer segments on revenue.

            Visualization:
                    Create scatter plots, heatmaps, bar charts, and other appropriate visualizations to support the analysis and highlight key insights.
                    Key Insights and Recommendations:

            Detail Analysis:
                    For columns having text values do a detail analysis of the contents.

        Summarize the most significant findings from the analysis.
        Provide actionable recommendations based on the data insights.
        Ensure the analysis is thorough, well-organized, and presented clearly. Use visualizations to support and illustrate key points wherever possible.

"""



def main():

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.set_page_config(page_title="Scolo", page_icon=":moneybag:")
    st.markdown("""
        <style>
        .subtitle {
            font-size: 20px;
            color: #555555;
        }
        </style>
        """, unsafe_allow_html=True)
    st.header("🦜🔗 AI Structured Data Analyst")
    st.markdown('<p class="subtitle">Upload a structured data file with any of the following extensions:[ .xlsx, .csv, .json, .parquet, .h5, .feather, .html ] . For optimal results, \
                the first line should be a header record. For Excel files, the current version of the app only supports single sheet. \
                Rest assured, your documents are read locally on your device, ensuring strict privacy compliance.\
                For better performance pls select one option at a time.</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose any file having extension  .xlsx, .csv, .json, .parquet, .h5, .feather, .html")

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
            try:
                file_extension = pathlib.Path(uploaded_file.name).suffix
                unique_filename = str(uuid.uuid4()) + file_extension
                uploads_folder = base_dir
                os.makedirs(uploads_folder, exist_ok=True)  # Create 'uploads' folder if it doesn't exist
                file_path = os.path.join(uploads_folder, unique_filename)
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())

                if file_extension == '.csv':
                    df = pd.read_csv(uploaded_file)
                elif file_extension == '.xlsx':
                    df = pd.read_excel(uploaded_file)
                elif file_extension == '.json':
                    df = pd.read_json(uploaded_file)
                elif file_extension == '.parquet':
                    df = pd.read_parquet(uploaded_file)
                elif file_extension == '.h5':
                    df = pd.read_hdf(uploaded_file)
                elif file_extension == '.feather':
                    df = pd.read_feather(uploaded_file)
                elif file_extension == '.html':
                    dfs_html = pd.read_html(uploaded_file)
                    df = dfs_html[0]
            except:
                st.error("Error in uploaded file, pls try with different file..")
                st.stop()



        if st.checkbox("Show top five records from uploaded file"):
            with st.expander('Data'):
                st.table(df.head(5).style.set_properties(**{'text-align': 'center'}).set_table_styles([{
                    'selector': 'th',
                    'props': [('text-align', 'center')]
                }]))



        if st.checkbox(" 📃  Generate a Comprehensive Analysis for uploaded Structured Data"):
            with st.spinner(text="Analysing document..."):
                with st.expander('AI Suggested reports'):
                    with st.spinner(text="Genertaing reports..."):
                        try:  
                            summary = get_spreadsheet_summary(df, summary_prompt)
                            st.info(summary)
                            st.success("Done!")
                        except:
                            st.error("Please refresh the URL link and try again..")
                            st.stop()

        if st.checkbox(" 📊 Generate Graphs on uploaded Structured Data"):
            column_names = ", ".join(df.columns)
            try:
                # Check if the uploaded DataFrame is not empty
                if not df.empty:
                    # Handle the OpenAI query and display results
                    handle_openai_query(df, column_names)
                else:
                    # Display a warning if the uploaded data is empty
                    st.warning("The given data is empty.")
            except:
                st.error("Sorry I can't understand your request, Please reformat your query and submit it again..")
                st.stop()
        
        if st.checkbox("👨‍💻 Chat with your uploaded Structured Data "):
            query = st.text_area("Insert your query")
            if st.button("Submit Query", type="primary"):
                try:
                    query = "using python_repl_ast, from the dataframe  " + query 
                    result = get_spreadsheet_summary(df, query)
                    st.info(result)
                    st.success("Done!")
                except:
                    st.error("Sorry I can't understand your request, Please reformat your query and submit it again..")
                    st.stop()

    

if __name__ == "__main__":
    main()