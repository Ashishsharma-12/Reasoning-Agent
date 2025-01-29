from reasoning_agent import create_reasoning_agent
from langchain_ollama import OllamaLLM

model = OllamaLLM(model="mistral:latest")

question = '''
Question: Write a function to detect a cycle in a directed graph.
graph = {
    0: [1],
    1: [2],
    2: [3],
    3: [1] 
}
'''
my_agent = create_reasoning_agent(model)

response = my_agent.invoke({"input": question})

print(response['output'])