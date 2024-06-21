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
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

import matplotlib.pyplot as plt
from langchain.agents.agent_types import AgentType
from contextlib import contextmanager

import streamlit as st
import streamlit_authenticator as stauth

import yaml
from yaml.loader import SafeLoader

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="Scolo", page_icon=":moneybag:")

with open('./config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
)
# import openai
from openai import OpenAI

# Define the model to use
MODEL_NAME = "gpt-3.5-turbo-0613"
# MODEL_NAME = "gpt-4-turbo"



llm = ChatOpenAI(temperature=0, model=MODEL_NAME)

client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY"),
)


def get_spreadsheet_summary(df, query):
    agent = create_pandas_dataframe_agent(
    # ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
    ChatOpenAI(temperature=0, model=MODEL_NAME),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    max_execution_time=None,
    max_iterations=120,
    handle_parsing_errors= True
    )
    return agent.run(query)


text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=10000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

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
            - Try to use the following color palette for coloring the plots : ["#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#ffee65", "#beb9db", "#fdcce5", "#8bd3c7"]
            
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

CSV_PROMPT_PREFIX = """
You are an agent designed to interact with a dataframe.
"""

CSV_PROMPT_SUFFIX = """
- **ALWAYS** before giving the Final Answer, try another method.
Then reflect on the answers of the two methods you did and ask yourself
if it answers correctly the original question.
If you are not sure, try another method.
- If the methods tried do not give the same result,reflect and
try again until you have two methods that have the same result.
- If you still cannot arrive to a consistent result, say that
you are not sure of the answer.
- If you are sure of the correct answer, create a beautiful
and thorough response using Markdown.
- **DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE,
ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE**.
- **ALWAYS**, as part of your "Final Answer", explain how you got
to the answer on a section that starts with: "\n\nExplanation:\n".
In the explanation, mention the column names that you used to get
to the final answer.
"""



map_prompt_template = """
                      Write a summary of the text fields and latent meaning of structured data present in text.
                      {text}
                      """

map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

combine_prompt_template = """
                      Write a concise summary of the following text delimited by triple backquotes.
                      Return your response in bullet points which covers the latent meaning present in text and avoild numbers in final report .
                      ```{text}```
                      BULLET POINT SUMMARY:
                      """

combine_prompt = PromptTemplate(
    template=combine_prompt_template, input_variables=["text"]
)

map_reduce_chain = load_summarize_chain(
    llm,
    chain_type="map_reduce",
    map_prompt=map_prompt,
    combine_prompt=combine_prompt,
    return_intermediate_steps=True,
)



def main():

    authenticator.login()
    if st.session_state["authentication_status"]:
        authenticator.logout()
        st.write(f'Welcome *{st.session_state["name"]}*')

        st.markdown("""
            <style>
            .subtitle {
                font-size: 20px;
                color: #555555;
            }
            </style>
            """, unsafe_allow_html=True)
        st.header("🦜🔗 AI Structured Data Analyst")
        st.markdown('<p class="subtitle">Upload a structured data file with any of the following extensions:[ .xlsx, .csv ] . For optimal results, \
                    the first line should be a header record. For Excel files, the current version of the app only supports single sheet. \
                    Rest assured, your documents are read locally on your device, ensuring strict privacy compliance.\
                    For better performance pls select one option at a time.</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose any file having extension  .xlsx, .csv")

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
                    unique_number = str(uuid.uuid4())
                    unique_filename = unique_number  + file_extension
                    xlsx_filename = unique_number  + '.xlsx'
                    uploads_folder = base_dir
                    os.makedirs(uploads_folder, exist_ok=True)  # Create 'uploads' folder if it doesn't exist
                    file_path = os.path.join(uploads_folder, unique_filename)
                    xlsx_path = os.path.join(uploads_folder, xlsx_filename)
                    with open(file_path, 'wb') as f:
                        f.write(uploaded_file.getvalue())

                    if file_extension == '.csv':
                        with st.spinner(text="Loading document..."):
                            df = pd.read_csv(uploaded_file)
                            df.head(1000).to_excel(xlsx_path,index=None, header=True)
                            loader = UnstructuredExcelLoader(xlsx_path)
                            data = loader.load()
                    elif file_extension == '.xlsx':
                        with st.spinner(text="Loading document..."):
                            df = pd.read_excel(uploaded_file)
                            loader = UnstructuredExcelLoader(xlsx_path)
                            data = loader.load()
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



            if st.checkbox(" 📃  Generate a Comprehensive Analysis for uploaded Structured Data"):
                with st.spinner(text="Analysing data it may take few mints pls wait..."):
                    with st.expander('AI Suggested Detail Analysis Reports'):
                        with st.spinner(text="Genertaing reports ..."):
                            try:  
                                texts = text_splitter.create_documents([data[0].dict()['page_content']])
                                map_reduce_outputs = map_reduce_chain({"input_documents": texts[:10]})
                                st.info(map_reduce_outputs['output_text'])
                                st.success("Done!")
                            except :
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
                input = st.text_area("Insert your query")
                if st.button("Submit Query", type="primary"):
                    try:
                        # query = "using python_repl_ast, from the dataframe  " + query + " avoid programming code as output"
                        query = (CSV_PROMPT_PREFIX + input + CSV_PROMPT_SUFFIX)
                        result = get_spreadsheet_summary(df, query)
                        st.info(result)
                        st.success("Done!")
                    except:
                        st.error("Sorry I can't understand your request, Please reformat your query and submit it again..")
                        st.stop()

    elif st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] is None:
        st.warning('Please enter your username and password')

if __name__ == "__main__":
    main()