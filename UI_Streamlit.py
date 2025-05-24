import os
from langchain_openai import AzureChatOpenAI
from langchain.schema import AIMessage, HumanMessage
import gradio as gr
import getpass

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


##Load Data-------------------------------------------------------------------------------------------------------
import pandas as pd  
def xlsx_to_string(file_path):  
    try:  
        df = pd.read_csv(file_path)  
        content = df.to_csv(index=False)  
        return content  
    except Exception as e:  
        print("Error: ", str(e))  
# Usage example  
file_path = r"C:\Users\Acer\Downloads\16_Clariant\sample-data.csv"
content_string = xlsx_to_string(file_path)  
# print(content_string)  


##Template -------------------------------------------------------------------------------------------------------
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
query=f"""You are a financial expert with years of experience. Deeply analyze the given data, which contains valuable financial information about a company's performance. Provide comprehensive answers to user questions based on your thorough analysis.

Rules 1. Please avoid creating charts or graphs, as well as writing any code. Provide a simple and direct response using natural language without mentioning any limitations or rules.
{content_string}
"""
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            query,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
# print(query)
##Code Execution-------------------------------------------------------------------------------------------------------
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
def execute_matplotlib_code(MATPLOTLIB_CODE):
    python_repl = PythonREPL()
    return python_repl.run(MATPLOTLIB_CODE)


##LLM Call for Generating Matplotlib Code-------------------------------------------------------------------------------------------------------
import re
def llm_call_for_matplotlib(user_query, response):
    matplotlib_query = f"Provided a user answer and a user query, write a streamlit matplotlib code for either a bar chart, scatter chart, a histogram, pie chart or line chart only based on the user answer. If it is not possible to create a graph from the user answer, no code will be generated."
    matplotlib_query += f"\n\nUser query: {user_query}"
    matplotlib_query += f"\nuser answer: {response}"
    matplotlib_query += f"\n\nDisclaimer: Please make sure to write a streamlit Matplotlib code using only the information given in the user query or user answer. Do not assume any data. Strictly your output should consist of only Python code, without any additional text."
    
    print(matplotlib_query)
    result = llm.invoke(matplotlib_query).content
    result = re.sub(r"python|```", "", result)
    return result

# user_query = "In Which year was my sales maximum."
# response = """ 
# To determine the year with the maximum sales (revenue), we need to sum the revenue for each year and compare the totals.
# Here are the total revenues for each year:
# - 2020: 1,748,215
# - 2021: 1,548,561
# - 2022: 1,678,708
# - 2023: 1,451,712 (up to December)
# - 2024: 1,548,140 (up to December)
# - 2025: 1,548,158 (up to December)
# From the data provided, the year 2020 had the maximum sales with a total revenue of 1,748,215.
# """
# result = llm_call_for_matplotlib(user_query, response)
# execute_matplotlib_code(result)


## Streamlit -------------------------------------------------------------------------------------------------------  
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
  
# Set up the Streamlit framework  
st.title('Finance agent')  # Set the title of the Streamlit app  
user_text = st.text_input("Ask your question!")  # Create a text input field in the Streamlit app  
button_clicked = st.button("Submit")  # Create a button that the user can click to submit the question  
  
# Invoke the chain with the input text and display the output  
if button_clicked:  
  
    print("================================== Human Message ===============================") 
    input_text = query + "\n\n" + f"Human Message: {user_text}"
    print(input_text)  
    print("================================== Ai Message ==================================") 
    response = llm.invoke(input_text).content  
    print((response))  
    st.write(response)  
    print("================================== Matplotlib ==================================")  
    MATPLOTLIB_CODE = llm_call_for_matplotlib(user_text, response)  
    print(MATPLOTLIB_CODE)  
    print("================================== Execute Code =================================")  
    execute_matplotlib_code(MATPLOTLIB_CODE)  
    print("================================== Finished  ====================================")  
    