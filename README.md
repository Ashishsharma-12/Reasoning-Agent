# Reasoning Agent with Reinforcement Learning (RL)

This project implements a powerful reasoning agent that uses a Transformer-based model (such as GPT) to analyze complex queries, break them down into logical components, retrieve relevant information, and generate well-reasoned responses. The agent follows the **ReAct (Reasoning + Acting)** framework, enabling it to reason, act, observe, and refine its outputs iteratively.

### Key Features:
- **Step-by-Step Reasoning**: Breaks down queries into logical steps.
- **Tool Use**: Utilizes various external tools like web search, data analysis, sentiment analysis, code execution, and reinforcement learning.
- **Self-Correction**: Can refine its response based on new insights from actions taken during reasoning.
- **Reinforcement Learning**: Implements a Q-learning-based reinforcement learning model to solve problems iteratively.
  
### Tools Integrated:
- **Web Search**: Uses DuckDuckGo to search for relevant information on the web.
- **Web Scraping**: Scrapes data from web pages for structured information.
- **Sentiment Analysis**: Analyzes the sentiment (polarity and subjectivity) of text using TextBlob.
- **Code Execution**: Dynamically executes Python code for performing data analysis and computations.
- **Q-learning**: Implements Q-learning for reinforcement learning tasks to solve problems with iterative improvement.
- **Task Planning**: Plans tasks and provides time estimations based on input.

### Technologies Used:
- **Python**: The primary language for implementation.
- **LangChain**: For managing agent flows and integrating various tools.
- **DuckDuckGo API**: For web search functionality.
- **TextBlob**: For performing sentiment analysis on text.
- **NumPy**: Used for Q-learning and mathematical computations.
- **BeautifulSoup**: For scraping data from web pages.

### Prerequisites:

Before you begin, make sure you have the following installed:
- Python 3.7 or above
- Required Python libraries (listed below)

### Installation:
1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/your-username/reasoning-agent.git
    cd reasoning-agent
    ```

2. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use venv\Scripts\activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up the environment variables:
    You can use a `.env` file to set up necessary environment variables like API keys for tools (DuckDuckGo, etc.), if applicable.

### Usage:

1. **Run the agent:**

    You can interact with the agent by initializing it in your code:
    ```python
    from reasoning_agent.agent import create_reasoning_agent_with_rl
    from langchain_core.prompts import PromptTemplate
    from langchain.agents import AgentExecutor

    # Initialize the agent with a pre-trained model (mock model for testing purposes)
    agent = create_reasoning_agent_with_rl(MockModel())
    
    # Execute a query:
    query = "What is the capital of France?"
    result = agent.execute(query)
    print(result)
    ```

2. **Run Tests:**

    To make sure everything is working as expected, you can run the test suite:
    ```bash
    pytest tests/test_agent.py
    ```

3. **Use Reinforcement Learning:**

    The agent can utilize Q-learning for reinforcement learning tasks. Simply query the agent with a task related to RL, such as:
    ```python
    query = "Use Q-learning to solve a simple maze problem."
    result = agent.execute(query)
    print(result)
    ```

### Customizing the Agent:
You can modify the agent by:
- Adding new tools to the reasoning process.
- Updating the model and prompt template to integrate with a different LLM (e.g., GPT-3, GPT-4, etc.).
- Adjusting the behavior of the tools for specific tasks.

### Example Queries:
- "What is the current weather in Paris?"
- "Analyze the sentiment of the text: 'I love programming!'"
- "Scrape the Wikipedia page for Python programming."
- "Use Q-learning to solve a simple maze problem."
- "Write Python code to calculate the factorial of a number."

### Tests:

The project includes tests for verifying the functionality of each tool and the reasoning process. To run the tests:

```bash
pytest tests/test_agent.py
