import autogen
import streamlit as st

from autogen.agentchat.contrib.agent_builder import AgentBuilder

config_file_or_env = "OAI_CONFIG_LIST.json"
llm_config = {"temperature": 0}
config_list = autogen.config_list_from_json(config_file_or_env, filter_dict={"model": ["gpt-4-1106-preview",
                                                                                       "gpt-3.5-turbo-16k"]})


# Step 1: prepare configuration and some useful functions
def start_task(execution_task: str, agent_list: list):
    group_chat = autogen.GroupChat(agents=agent_list, messages=[], max_round=12)
    manager = autogen.GroupChatManager(groupchat=group_chat, llm_config={"config_list": config_list, **llm_config})
    agent_list[0].initiate_chat(manager, message=execution_task)


# Step 2: create a AgentBuilder
builder = AgentBuilder(
    config_file_or_env=config_file_or_env,
    builder_model="gpt-4-1106-preview",
    agent_model="gpt-4-1106-preview"
)

scenario_list = ["Generate some agents to analyse latest stock price performance  from Yahoo finance and recommend if "
                 "a given stock is a buy, sell or a hold",
                 "Generate some agents that can find papers on arxiv by programming and analyzing them in specific "
                 "domains related to generative ai and deep learning"]

st.header("Automatic Agent Creator !!")
st.subheader("This is a demo of how to use the autogen library to create agents that can perform tasks in a group chat")
building_task = st.selectbox("Enter the building task", scenario_list)
building_task_desc = st.text_input("Enter the task you want the agents to perform")
submit = st.button("submit", type="primary")

if building_task and building_task_desc and submit:
    agent_list, agent_configs = builder.build(building_task, llm_config)
    start_task(
        execution_task=building_task_desc,
        agent_list=agent_list,
    )
    builder.clear_all_agents(recycle_endpoint=True)
