from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from typing import TypedDict, Dict, List, Optional, Literal, Annotated, Any
from pydantic import BaseModel, Field
from prompts import OMISSION_PROMPT, WORD_RPLC_PROMPT, WORD_ORDR_PROMPT, WORD_SYNM_PROMPT, FLUENT_PROMPT, MIXD_LANG_PROMPT
from dotenv import load_dotenv
import os
load_dotenv()


def merge_dicts(left: dict, right: dict) -> dict:
    merged = dict(left or {})
    merged.update(right or {})
    return merged

class AgentOutput(BaseModel):
    probability: float = Field(..., description="The probability that the error is present", ge=0.0, le=1.0)
    evidence: str = Field(..., description="Brief explanation pointing to concrete words, phrases, or structures that justify the probability.")
    confidence: Literal["low", "medium", "high"] = Field(..., description="Model confidence in this judgment. Must be one of: low, medium, high.")
    possible_overlap: List[str] = Field(..., description="List of other error categories that may overlap or influence this error judgment.")

class MTState(TypedDict):
    source: str
    mt: str
    reference: str
    
    omission: Optional[AgentOutput]
    mixd_lang: Optional[AgentOutput]
    word_ordr: Optional[AgentOutput]
    word_rplc: Optional[AgentOutput]
    word_synm: Optional[AgentOutput]
    fluent: Optional[AgentOutput]

    agent_reports: Annotated[Dict[str, Any], merge_dicts]
    round: int
    max_rounds: int

    final_output: Optional[List]

# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def make_error_agent(system_prompt: str, state_key: str):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """
SOURCE SENTENCE: {source}
         
MACHINE TRANSLATED SENTENCE: {translated}
         
REFERENCE SENTENCE: {reference}
         
Current round: {round}
         
Other agent findings so far: {agent_reports}
         
If this is round > 1:
- Re-evaluate your probability based on other agent reports and also by re-evaluating the translation.
- Adjust only if justified.
- Explain change briefly
""")
    ])

    chain = prompt_template | llm.with_structured_output(AgentOutput)

    def agent_fn(state: MTState) -> Dict[str, AgentOutput]:
        output = chain.invoke({
            "source": state["source"],
            "translated": state["mt"],
            "reference": state["reference"],
            "agent_reports": state.get("agent_reports", {}),
            "round": state["round"]
        })

        updated_reports = dict(state.get("agent_reports", {}))
        updated_reports[state_key] = output.model_dump()

        return {state_key: output, "agent_reports": updated_reports}
    return agent_fn

def loop_controller(state: MTState):
    if state["round"] < state["max_rounds"]:
        return {"round": state["round"] + 1}
    return {}

omission_agent = make_error_agent(OMISSION_PROMPT, "omission")
mixd_lang_agent = make_error_agent(MIXD_LANG_PROMPT, "mixd_lang")
word_ordr_agent = make_error_agent(WORD_ORDR_PROMPT, "word_ordr")
word_rplc_agent = make_error_agent(WORD_RPLC_PROMPT, "word_rplc")
word_synm_agent = make_error_agent(WORD_SYNM_PROMPT, "word_synm")
fluent_agent = make_error_agent(FLUENT_PROMPT, "fluent")

def aggregate(state: MTState) -> Dict[str, List]:
    return {"final_output": [state["omission"].probability, state["mixd_lang"].probability, state["word_ordr"].probability, state["word_rplc"].probability, state["word_synm"].probability, state["fluent"].probability],
            "agent_reports": state["agent_reports"]}

graph = StateGraph(MTState)

graph.add_node("omission", omission_agent)
graph.add_node("mixd_lang", mixd_lang_agent)
graph.add_node("word_ordr", word_ordr_agent)
graph.add_node("word_rplc", word_rplc_agent)
graph.add_node("word_synm", word_synm_agent)
graph.add_node("fluent", fluent_agent)
graph.add_node("aggregate", aggregate)
graph.add_node("loop_controller", loop_controller)

graph.set_entry_point("omission")

graph.add_edge("omission", "mixd_lang")
graph.add_edge("omission", "word_ordr")
graph.add_edge("omission", "word_rplc")
graph.add_edge("omission", "word_synm")
graph.add_edge("omission", "fluent")

graph.add_edge("fluent", "loop_controller")

def should_continue(state: MTState):
    if state["round"] < state["max_rounds"]:
        return "continue"
    return "done"

graph.add_conditional_edges("loop_controller", should_continue, 
                            {
                                "continue": "omission",
                                "done": "aggregate"
                            })

# graph.add_edge("mixd_lang", "aggregate")
# graph.add_edge("word_ordr", "aggregate")
# graph.add_edge("word_rplc", "aggregate")
# graph.add_edge("word_synm", "aggregate")
# graph.add_edge("fluent", "aggregate")

graph.add_edge("aggregate", END)

app = graph.compile()

if __name__ == "__main__":
    intput_state = {
        "source": "The qualities that determine a subculture as distinct may be linguistic, aesthetic, religious, political, sexual, geographical, or a combination of factors.",
        "mt": "उपसंस्कृति को विशिष्ट रूप से निर्धारित करने वाले गुण भाषाई, सौंदर्य, धार्मिक, राजनीतिक, यौन, भौगोलिक या कारकों का संयोजन हो सकते हैं।",
        "reference": "वे गुण जो किसी उप-संस्कृति को अलग बनाते हैं, जैसे कि भाषा, सौंदर्य, धर्म, राजनीति, यौन, भूगोल या कई सारे कारकों का मिश्रण हो सकते हैं.",
        "agent_reports": {},
        "round": 1,
        "max_rounds": 2
    }

    result = app.invoke(intput_state)
    print(result)
