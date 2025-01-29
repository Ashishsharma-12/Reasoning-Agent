from dotenv import load_dotenv, find_dotenv
from langchain.agents.react.agent import create_react_agent
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain.agents import Tool
import pandas as pd
import numpy as np
from langchain_core.prompts.prompt import PromptTemplate
from typing import Optional, Any, Dict
from langchain.agents.agent import AgentExecutor
from langchain_core.utils.interactive_env import is_interactive_env
from bs4 import BeautifulSoup
import requests
from textblob import TextBlob

# print("library loaded")

load_dotenv(find_dotenv())

# # Agent function

class RL_Environment:
    def __init__(self):
        self.state_space_size = 5  # Number of possible states
        self.action_space = [0, 1, 2]  # Example action space: [0, 1, 2] (3 possible actions)
        self.state = 0  # Start at state 0
    
    def reset(self):
        self.state = 0  # Reset to initial state
        return self.state
    
    def step(self, action):
        # Logic for transitioning between states and giving rewards
        if action == 0:
            reward = 1  # Positive reward for action 0
            next_state = 1
        elif action == 1:
            reward = -1  # Negative reward for action 1
            next_state = 2
        else:
            reward = 0  # Neutral reward for action 2
            next_state = 3
        
        done = next_state == 4  # End the episode if we reach state 4
        return next_state, reward, done


def create_reasoning_agent(model):

    # Define Tools

    search = DuckDuckGoSearchResults()

    search_tool = [
        Tool(
            name = "search",
            func=search.run,
            description="useful for when you need to answer questions using web search. You should ask targeted questions"
        )
    ]

    def scrape_webpage(url: str):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.prettify()

    web_scraping_tool = [
        Tool(
            name="scrape_webpage",
            func=scrape_webpage,
            description="Useful for extracting structured data from web pages."
        )
    ]


    def analyze_sentiment(text: str):
        blob = TextBlob(text)
        sentiment = blob.sentiment
        return sentiment

    sentiment_analysis_tool = [
        Tool(
            name="analyze_sentiment",
            func=analyze_sentiment,
            description="Useful for analyzing the sentiment of a given text input."
        )
    ]

    def execute_code(input_code: str):
        try:
            exec(input_code)
            return locals()
        except Exception as e:
            return str(e)

    code_execution_tool = [
        Tool(
            name="execute_code",
            func=execute_code,
            description="Executes Python code dynamically. Useful for running data analysis and performing computations."
        )
    ]

    def q_learning(environment, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
        q_table = np.zeros((environment.state_space_size, len(environment.action_space)))
        
        for _ in range(episodes):
            state = environment.reset()
            done = False
            while not done:
                if np.random.rand() < epsilon:
                    action = np.random.choice(environment.action_space)
                else:
                    action = np.argmax(q_table[state])  # Select the best action based on Q-table
                    
                next_state, reward, done = environment.step(action)
                # Update Q-value using the Q-learning equation
                q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]))
                state = next_state  # Transition to the next state
        
        return q_table

    rl_environment = RL_Environment()
    
    rl_tool = [
        Tool(
            name="q_learning",
            func=lambda: q_learning(rl_environment),
            description="Uses Q-learning for decision-making and learning."
        )
    ]
    
    def plan_task(task_name: str, time_estimation: int):
        return f"Task '{task_name}' planned for {time_estimation} hours."

    task_planning_tool = [
        Tool(
            name="plan_task",
            func=plan_task,
            description="Useful for planning and organizing tasks based on given inputs."
        )
    ]

    tools = search_tool + web_scraping_tool + sentiment_analysis_tool + code_execution_tool + rl_tool + task_planning_tool

    # Creating environment

    if is_interactive_env():
        pd.set_option("display.max_columns", None)
    
    # Query template

    PREFIX = """
    You are an advanced reasoning agent powered by a Transformer-based model.
    Your primary goal is to analyze complex queries, break them into logical steps,
    retrieve relevant information, and generate well-reasoned responses.

    You follow the ReAct (Reasoning + Acting) framework, which involves:
    1. Thinking – Analyzing the question, breaking it down into smaller, logical components.
    2. Acting – Choosing appropriate tools (e.g., web search, code execution, data analysis) to gather the necessary information.
    3. Observing – Assessing the results of your actions.
    4. Self-Correction – Refining your reasoning and response based on new insights from the observations.

    You have access to various reasoning tools, such as:
    - Logical deduction
    - Mathematical computation
    - Data retrieval (if available)
    - Code execution (if enabled)

    Follow the structured reasoning process below to answer the given question.
    """

    FORMAT_INSTRUCTIONS = """
    Use the following structured format for reasoning:

    Question: The input query that requires reasoning
    Thought: Analyze the question and break it into logical components
    Action: The action to take, should be one of [{tool_names}]
    Action Input: The input parameters for the chosen action
    Observation: The result or retrieved information from the action
    ... (This Thought/Action/Action Input/Observation cycle can repeat multiple times)
    Thought: I now have enough information to conclude
    Final Answer: The well-reasoned answer to the original query. If the Final Answer is provided, stop further iterations.
    """

    SUFFIX = """
    Now, begin your reasoning process!

    Question: {input}
    Thought:{agent_scratchpad}
    """

    template  = "".join([PREFIX, "\n{tools}\n", FORMAT_INSTRUCTIONS, SUFFIX])

    prompt = PromptTemplate.from_template(template)

    # Initialize agent

    agent = create_react_agent(llm = model, tools=tools, prompt=prompt)

    agent_executor_kwargs : Optional[Dict[str, Any]] = None

    rs_agent = AgentExecutor(
            agent=agent,
            tools=tools,
            callback_manager=None,
            verbose=True,
            return_intermediate_steps=True,
            max_iterations=15,
            max_execution_time=None,
            early_stopping_method="force",
            handle_parsing_errors=True,
            **(agent_executor_kwargs or {}),
        )
    
    return rs_agent


























