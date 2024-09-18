import os
from dotenv import load_dotenv
import openai
import requests
import json

import time
import logging
from datetime import datetime
# import streamlit as st


load_dotenv()

client = openai.OpenAI()

model = "gpt-4o"

# Step 1. Upload a file to OpenaI embeddings ===
filepath = "./spacescience.pdf"    
file_streams = [open(filepath, "rb")]

# Create a vector store caled "Financial Statements"
vector_store = client.beta.vector_stores.create(name="space Information")

file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
  vector_store_id=vector_store.id, files=file_streams
)
 
# You can print the status and the file counts of the batch to see the result of this operation.
print(file_batch.status)
print(file_batch.file_counts)

# Step 2 - Create an assistant 
assistant = client.beta.assistants.create(
    name="space Educator",  
    instructions="""You are a knowledgeable assistant specializing in space and astronomy.   
    Your role is to provide summaries of space-related research papers, clarify terminology within the context of astronomy, and extract key figures and data from space missions and discoveries. 
    Cross-reference information to offer additional insights and answer related questions comprehensively. 
    Analyze space research papers, noting their strengths and limitations. You should be able to take a list of article titles and descriptions and use them to provide accurate and comprehensive responses about cosmic phenomena and space exploration. 
    Respond to queries effectively, incorporating user feedback to enhance accuracy. 
    Handle space data securely and update your knowledge base with the latest astronomical research and discoveries. 
    Adhere to ethical standards, respect intellectual property, and guide users on any limitations. 
    Maintain a feedback loop for continuous improvement and user support. 
    Your ultimate goal is to facilitate a deeper understanding of the universe, making complex space material more accessible and engaging.""", 
    tools=[{"type": "code_interpreter"},{"type": "file_search"},  
            {"type": "function",  
                "function": {
                    "name": "get_news",
                    "description": "Get the list of articles/news for the given topic",   
                    "parameters": {
                        "type": "object",
                        "properties": { 
                            "topic": {
                                "type": "string",
                                "description": "The topic for the news, e.g.space ",
                            }
                        },
                        "required": ["topic"],
                    },
                },
            }
    ],
        model=model,
)

assistant = client.beta.assistants.update(
  assistant_id=assistant.id,
  tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
)

# === Get the Assis ID ===
assis_id = assistant.id
print(assis_id)


# == Step 3. Create a Thread

thread = client.beta.threads.create()
thread_id = thread.id
print(thread_id)

# # == Hardcoded ids to be used once the first code run is done and the assistant was created
# assis_id = "asst_dMnnWRgC7k81opPNxLt7vH5I"
# thread_id = "thread_CRLCFmnVPNbQaMHyu1RfJbSj"

# message = "What is mining? answer according to the resourses you are trained for!"
# message = client.beta.threads.messages.create(
#     thread_id=thread_id, role="user", content=message
# )

# # == Run the Assistant
# run = client.beta.threads.runs.create(
#     thread_id=thread_id,
#     assistant_id=assis_id,
#     instructions="Please address the user as Bruce",
# )


# def wait_for_run_completion(client, thread_id, run_id, sleep_interval=5):
#     """
#     Waits for a run to complete and prints the elapsed time.:param client: The OpenAI client object.
#     :param thread_id: The ID of the thread.
#     :param run_id: The ID of the run.
#     :param sleep_interval: Time in seconds to wait between checks.
#     """
#     while True:
#         try:
#             run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
#             if run.completed_at:
#                 elapsed_time = run.completed_at - run.created_at
#                 formatted_elapsed_time = time.strftime(
#                     "%H:%M:%S", time.gmtime(elapsed_time)
#                 )
#                 print(f"Run completed in {formatted_elapsed_time}")
#                 logging.info(f"Run completed in {formatted_elapsed_time}")
#                 # Get messages here once Run is completed!
#                 messages = client.beta.threads.messages.list(thread_id=thread_id)
#                 last_message = messages.data[0]
#                 response = last_message.content[0].text.value
#                 print(f"Assistant Response: {response}")
#                 break
#         except Exception as e:
#             logging.error(f"An error occurred while retrieving the run: {e}")
#             break
#         logging.info("Waiting for run to complete...")
#         time.sleep(sleep_interval)

 
# # == Run it
# wait_for_run_completion(client=client, thread_id=thread_id, run_id=run.id)
 
# # === Check the Run Steps - LOGS ===
# run_steps = client.beta.threads.runs.steps.list(thread_id=thread_id, run_id=run.id)
# print(f"Run Steps --> {run_steps.data[0]}")