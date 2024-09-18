import asyncio
from thread_manager import ThreadManager
from assistant_manager import AssistantManager
import requests
import json


def get_news(topic):
    news_api_key = "09f0bf58f68445b79006faa012009b95"
    url = (
        f"https://newsapi.org/v2/everything?q={topic}&apiKey={news_api_key}&pageSize=5"
    )

    try:
        response = requests.get(url)
        print(response.status_code)
        if response.status_code == 200:
            news = json.dumps(response.json(), indent=4)
            news_json = json.loads(news)

            data = news_json

            # Access all the fiels == loop through
            status = data["status"]
            total_results = data["totalResults"]
            articles = data["articles"]
            final_news = []

            # Loop through articles
            for article in articles:
                source_name = article["source"]["name"]
                author = article["author"]
                title = article["title"]
                description = article["description"]
                url = article["url"]
                content = article["content"]
                title_description = f"""
                   Title: {title}, 
                   Author: {author},
                   Source: {source_name},
                   Description: {description},
                   URL: {url}
            
                """
                final_news.append(title_description)
            print(final_news)
            return final_news
        else:
            return []

    except requests.exceptions.RequestException as e:
        print("Error occured during API Request", e)


class ChatSession:
    def __init__(self, thread_manager: ThreadManager, assistant_manager: AssistantManager, assistant_name: str,
                 model_name: str, assistant_id: str = None, thread_id: str = None):
        self.thread_manager = thread_manager
        self.assistant_manager = assistant_manager
        self.assistant_name = assistant_name
        self.model_name = model_name
        self.assistant_id = assistant_id
        self.thread_id = thread_id

    async def start_session(self):
        if self.thread_id is None:
            # Get or create a thread
            self.thread_id = await self.get_or_create_thread()

        if self.assistant_id is None:
            # Find or create an assistant
            self.assistant_id = await self.find_or_create_assistant(
                name=self.assistant_name,
                model=self.model_name
            )

        # Display existing chat history
        await self.display_chat_history()

        prev_messages = await self.thread_manager.list_messages(self.thread_id)
        if prev_messages is None:
            print("An error occurred while retrieving messages.")
            return

        # Start the chat loop
        await self.chat_loop()

    async def chat_loop(self):
        try:
            while True:
                user_input = input("You: ")
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    break
                if user_input.lower() in ['/delete', '/clear']:
                    await self.thread_manager.delete_thread(self.thread_id)
                    self.thread_id = await self.get_or_create_thread()
                    continue

                response = await self.get_latest_response(user_input)

                if response:
                    print("Assistant:", response)

        finally:
            print(f"Session ended")

    async def get_or_create_thread(self):
        data = self.thread_manager.read_thread_data()
        thread_id = data.get('thread_id')
        if not thread_id:
            thread = await self.thread_manager.create_thread(messages=[])
            thread_id = thread.id
            self.thread_manager.save_thread_data(thread_id)
        return thread_id

    async def find_or_create_assistant(self, name: str, model: str):
        """
        Finds an existing assistant by name or creates a new one.

        Args:
            name (str): The name of the assistant.
            model (str): The model ID for the assistant.

        Returns:
            str: The ID of the found or created assistant.
        """
        assistant_id = await self.assistant_manager.get_assistant_id_by_name(name)
        if not assistant_id:
            # assistant = await self.assistant_manager.create_assistant(name=name,
            #                                                           model=model,
            #                                                           instructions="",
            #                                                           tools=[{"type": "retrieval"}]
            #                                                           )
            # assistant_id = assistant.id
            # Step 1. Upload a file to OpenaI embeddings ===
            filepath = "./spacescience.pdf"
            file_streams = [open(filepath, "rb")]

            # Create a vector store caled "Financial Statements"
            vector_store = await self.assistant_manager.beta.vector_stores.create(name="space Information")

            file_batch = await self.assistant_manager.beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id, files=file_streams
            )
            
            # You can print the status and the file counts of the batch to see the result of this operation.
            print(file_batch.status)
            print(file_batch.file_counts)

            # Step 2 - Create an assistant 
            assistant = await self.assistant_manager.beta.assistants.create(
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
                                            "description": "The topic for the news, e.g. space",
                                        }
                                    },
                                    "required": ["topic"],
                },
                },}],
                model=model,
            )

            assistant = await self.assistant_manager.beta.assistants.update(
            assistant_id=assistant.id,
            tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
            )

            # === Get the Assis ID ===
            assis_id = assistant.id
            print(assis_id)

        return assistant_id

    async def send_message(self, content):
        return await self.thread_manager.send_message(self.thread_id, content)

    async def display_chat_history(self):
        messages = await self.thread_manager.list_messages(self.thread_id)
        if messages is None:
            return
        print(messages)
        for message in reversed(messages.data):
            role = message.role
            content = message.content[0].text.value  # Assuming message content is structured this way
            print(f"{role.title()}: {content}")

    async def get_latest_response(self, user_input):
        # Send the user message
        await self.send_message(user_input)

        # Create a new run for the assistant to respond
        await self.create_run()

        # Wait for the assistant's response
        await self.wait_for_assistant()

        # Retrieve the latest response
        return await self.retrieve_latest_response()

    async def create_run(self):
        return await self.thread_manager.create_run(self.thread_id, self.assistant_id)

############################################################################################################
    async def call_required_functions(self,required_actions,run):
        
        if not run:
            return

        tool_outputs = []

        for action in required_actions["tool_calls"]:
            func_name = action["function"]["name"]
            arguments = json.loads(action["function"]["arguments"])

            if func_name == "get_news":
                output = get_news(topic=arguments["topic"])
                final_str = ""
                for item in output:
                    final_str += "".join(item)

                tool_outputs.append({"tool_call_id": action["id"], "output": final_str}) 
            else:
                raise ValueError(f"Unknown function: {func_name}")

        print("Submitting outputs back to the Assistant...")
        await self.assistant_manager.client.beta.threads.runs.submit_tool_outputs_and_poll(
            thread_id=self.thread_id, run_id=run.id, tool_outputs=tool_outputs
        )
############################################################################################################

    async def wait_for_assistant(self):
        while True:
            runs = await self.thread_manager.list_runs(self.thread_id)
            latest_run = runs.data[0]
            if latest_run.status in ["completed", "failed"]:
                break
            elif latest_run.status == "requires_action":
                await self.call_required_functions(
                        required_actions=latest_run.required_action.submit_tool_outputs.model_dump(),run = latest_run
                    )

            await asyncio.sleep(2)  # Wait for 2 seconds before checking again

    async def retrieve_latest_response(self):
        response = await self.thread_manager.list_messages(self.thread_id)
        for message in response.data:
            if message.role == "assistant":
                return message.content[0].text.value
        return None
 