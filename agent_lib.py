import json

import autogen
from autogen.agentchat.contrib.agent_builder import AgentBuilder

config_file_or_env = "OAI_CONFIG_LIST.json"  # modify path
llm_config = {"temperature": 0}
config_list = autogen.config_list_from_json(config_file_or_env, filter_dict={"model": ["gpt-4-1106-preview",
                                                                                       "gpt-3.5-turbo-16k"]})


def start_task(execution_task: str, agent_list: list):
    group_chat = autogen.GroupChat(agents=agent_list, messages=[], max_round=12)
    manager = autogen.GroupChatManager(groupchat=group_chat, llm_config={"config_list": config_list, **llm_config})
    agent_list[0].initiate_chat(manager, message=execution_task)


AGENT_SYS_MSG_PROMPT = """Considering the following position:

POSITION: {position}

What requirements should this position be satisfied?

Hint:
# Your answer should be in one sentence.
# Your answer should be natural, starting from "As a ...".
# People with the above position need to complete a task given by a leader or colleague.
# People will work in a group chat, solving tasks with other people with different jobs.
# The modified requirement should not contain the code interpreter skill.
# Coding skill is limited to Python.
"""

position_list = [
    "Software engineer",
    "software product manager",
    "software architect",
    "quality assurance engineer",
    "Data_Analyst",
    "Programmer",
    "IT_Specialist",
    "Machine_Learning_Engineer"]

build_manager = autogen.OpenAIWrapper(config_list=config_list)
sys_msg_list = []

for pos in position_list:
    resp_agent_sys_msg = (
        build_manager.create(
            messages=[
                {
                    "role": "user",
                    "content": AGENT_SYS_MSG_PROMPT.format(
                        position=pos,
                        default_sys_msg=autogen.AssistantAgent.DEFAULT_SYSTEM_MESSAGE,
                    ),
                }
            ]
        )
        .choices[0]
        .message.content
    )
    sys_msg_list.append({"name": pos, "profile": resp_agent_sys_msg})

json.dump(sys_msg_list, open("./agent_library_example.json", "w"), indent=4)

library_path_or_json = "./agent_library_example.json"
building_task = "Generate a team of agents to create python microservices."

new_builder = AgentBuilder(
    config_file_or_env=config_file_or_env, builder_model="gpt-4-1106-preview", agent_model="gpt-4-1106-preview"
)
agent_list, _ = new_builder.build_from_library(building_task, library_path_or_json, llm_config)
start_task(
    execution_task="generate fully functional microservice to manage trades in a stock exchange.",
    agent_list=agent_list,
)
new_builder.clear_all_agents()
