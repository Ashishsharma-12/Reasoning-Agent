import pytest
from reasoning_agent.agent import create_reasoning_agent_with_rl
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
import numpy as np

# Mock model for testing
class MockModel:
    def __init__(self):
        pass
    
    def __call__(self, input_str):
        return input_str

@pytest.fixture
def mock_agent():
    # Setup: Create a mock model and initialize the agent
    model = MockModel()
    agent = create_reasoning_agent_with_rl(model)
    return agent

def test_agent_initialization(mock_agent):
    # Test agent initialization
    assert mock_agent is not None
    assert isinstance(mock_agent, AgentExecutor), "Agent should be of type AgentExecutor"

def test_agent_execution_with_query(mock_agent):
    # Test if agent can process a basic query
    query = "What is the capital of France?"
    result = mock_agent.execute(query)
    assert result is not None
    assert "France" in result['output'], "Agent should find relevant information for the query"

def test_reinforcement_learning_tool(mock_agent):
    # Test the Q-learning tool
    query = "Use Q-learning to solve a simple problem."
    result = mock_agent.execute(query)
    assert result is not None
    assert isinstance(result['output'], np.ndarray), "The result should be a Q-table (np.ndarray)"

def test_search_tool(mock_agent):
    # Test the search tool
    query = "Search for information on quantum computing."
    result = mock_agent.execute(query)
    assert result is not None
    assert "quantum" in result['output'], "Search should return relevant results based on query"

def test_sentiment_analysis_tool(mock_agent):
    # Test sentiment analysis tool
    query = "I love programming!"
    result = mock_agent.execute(query)
    assert result is not None
    assert isinstance(result['output'], tuple), "Sentiment analysis should return a tuple (polarity, subjectivity)"
    assert result['output'][0] > 0, "Sentiment should be positive"

def test_code_execution_tool(mock_agent):
    # Test code execution tool
    query = "Execute Python code: x = 10; y = 20; x + y"
    result = mock_agent.execute(query)
    assert result is not None
    assert result['output'] == 30, "Code execution result should be 30"

def test_web_scraping_tool(mock_agent):
    # Test web scraping tool
    query = "Scrape the Wikipedia page for Python programming."
    result = mock_agent.execute(query)
    assert result is not None
    assert "Python" in result['output'], "Scraping should return relevant data containing 'Python'"

if __name__ == "__main__":
    pytest.main()
