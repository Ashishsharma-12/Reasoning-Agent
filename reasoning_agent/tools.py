from langchain.agents import Tool
import numpy as np
import requests
from textblob import TextBlob
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from .environment import RL_Environment
from bs4 import BeautifulSoup


def q_learning(environment, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    q_table = np.zeros((environment.state_space_size, len(environment.action_space)))
    for _ in range(episodes):
        state = environment.reset()
        done = False
        while not done:
            if np.random.rand() < epsilon:
                action = np.random.choice(environment.action_space)
            else:
                action = np.argmax(q_table[state])
                
            next_state, reward, done = environment.step(action)
            q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]))
            state = next_state
    
    return q_table

def create_tools():
    # Define all tools

    search = DuckDuckGoSearchResults()

    search_tool = Tool(
        name="search",
        func=search.run,
        description="Useful for answering questions via web search. Targeted queries are needed."
    )

    def scrape_webpage(url: str):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.prettify()

    web_scraping_tool = Tool(
        name="scrape_webpage",
        func=scrape_webpage,
        description="Extract structured data from web pages."
    )

    def analyze_sentiment(text: str):
        blob = TextBlob(text)
        sentiment = blob.sentiment
        return sentiment

    sentiment_analysis_tool = Tool(
        name="analyze_sentiment",
        func=analyze_sentiment,
        description="Analyze the sentiment of the provided text."
    )

    def execute_code(input_code: str):
        try:
            exec(input_code)
            return locals()
        except Exception as e:
            return str(e)

    code_execution_tool = Tool(
        name="execute_code",
        func=execute_code,
        description="Execute Python code dynamically. Useful for data analysis and computations."
    )

    rl_environment = RL_Environment()

    def q_learning():
        return q_learning(rl_environment)

    reinforcement_learning_tool = Tool(
        name="q_learning",
        func=q_learning,
        description="Use Q-learning to make decisions based on the environment."
    )

    return [search_tool, web_scraping_tool, sentiment_analysis_tool, code_execution_tool, reinforcement_learning_tool]



