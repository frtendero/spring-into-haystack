import os
from pprint import pprint

from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.components.agents import Agent
from haystack_integrations.tools.mcp import MCPTool, StdioServerInfo

import logging

logging.basicConfig(level=logging.INFO)


github_mcp_server = StdioServerInfo(
    command="docker",
    args=[
        "run",
        "-i",
        "--rm",
        "-e",
        "GITHUB_PERSONAL_ACCESS_TOKEN",
        "mcp/github-mcp-server",
    ],
    env={
        "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"),
    },
)

logging.info("MCP server is created")

tool_get_file_content = MCPTool(
    name="get_file_contents",
    server_info=github_mcp_server,
)
tool_create_issue = MCPTool(
    name="create_issue",
    server_info=github_mcp_server,
)
tool_list_issues = MCPTool(
    name="list_issues",
    server_info=github_mcp_server,
)

tools = [tool_get_file_content, tool_create_issue, tool_list_issues]

logging.info("MCP tools are created")
agent = Agent(
    chat_generator=OllamaChatGenerator(
        url=os.getenv("OLLAMA_API_BASE_URL"),
        model="qwen3:32b",
        timeout=600,
        generation_kwargs={"temperature": 0.1, "num_ctx": 120_000},
    ),
    system_prompt="""
        /no_think
        You are a helpful agent provided with tools to perform different tasks
        on the github repository https://github.com/frtendero/spring-into-haystack
        owned by user frtendero. You will be prompted to perform tasks such as:
        reading files on the repository, check for errors in the files,
        list, read and create issues, and other typical github tasks.
    """,
    tools=tools,
)

logging.info("Agent created")


user_input = """There is a typo in the following readme file
https://raw.githubusercontent.com/frtendero/spring-into-haystack/refs/heads/main/README.md
of the repository frtendero/spring-into-haystack. Read the raw markdown file in its entirty
and search for the hidden typo.
Then, create an issue in the repository with the title 'Typo in README.md'
and specify the typo in the issue description and label it as 'typo'"""

response = agent.run(messages=[ChatMessage.from_user(text=user_input)])

pprint(response)  # full response with thinking process
print(response["messages"][-1].text)


# Now let's list the issues on the repo
user_input = "List the active issues on the github repository"
response = agent.run(messages=[ChatMessage.from_user(text=user_input)])
pprint(response)  # full response with thinking process
print(response["messages"][-1].text)
