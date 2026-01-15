from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Dict, List, Optional, Any
from pydantic import BaseModel, Field
from prompts import OMISSION_PROMPT, WORD_RPLC_PROMPT, WORD_ORDR_PROMPT, WORD_SYNM_PROMPT, FLUENT_PROMPT, MIXD_LANG_PROMPT
from dotenv import load_dotenv
load_dotenv()

class AgentOutput(BaseModel):
    probability: float = Field(..., description="The probability that the error is present", ge=0.0, le=1.0)

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

    final_output: Optional[List]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def make_error_agent(system_prompt: str, state_key: str):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """
SOURCE SENTENCE: {source}
         
MACHINE TRANSLATED SENTENCE: {translated}
         
REFERENCE SENTENCE: {reference}
""")
    ])

    chain = prompt_template | llm.with_structured_output(AgentOutput)

    def agent_fn(state: MTState) -> Dict[str, AgentOutput]:
        output = chain.invoke({
            "source": state["source"],
            "translated": state["mt"],
            "reference": state["reference"]
        })

        return {state_key: output}
    return agent_fn


omission_agent = make_error_agent(OMISSION_PROMPT, "omission")
mixd_lang_agent = make_error_agent(MIXD_LANG_PROMPT, "mixd_lang")
word_ordr_agent = make_error_agent(WORD_ORDR_PROMPT, "word_ordr")
word_rplc_agent = make_error_agent(WORD_RPLC_PROMPT, "word_rplc")
word_synm_agent = make_error_agent(WORD_SYNM_PROMPT, "word_synm")
fluent_agent = make_error_agent(FLUENT_PROMPT, "fluent")

def aggregate(state: MTState) -> Dict[str, List]:
    return {"final_output": [state["omission"].probability, state["mixd_lang"].probability, state["word_ordr"].probability, state["word_rplc"].probability, state["word_synm"].probability, state["fluent"].probability]}

graph = StateGraph(MTState)

graph.add_node("omission", omission_agent)
graph.add_node("mixd_lang", mixd_lang_agent)
graph.add_node("word_ordr", word_ordr_agent)
graph.add_node("word_rplc", word_rplc_agent)
graph.add_node("word_synm", word_synm_agent)
graph.add_node("fluent", fluent_agent)
graph.add_node("aggregate", aggregate)

graph.set_entry_point("omission")

graph.add_edge("omission", "mixd_lang")
graph.add_edge("omission", "word_ordr")
graph.add_edge("omission", "word_rplc")
graph.add_edge("omission", "word_synm")
graph.add_edge("omission", "fluent")

graph.add_edge("mixd_lang", "aggregate")
graph.add_edge("word_ordr", "aggregate")
graph.add_edge("word_rplc", "aggregate")
graph.add_edge("word_synm", "aggregate")
graph.add_edge("fluent", "aggregate")

graph.add_edge("aggregate", END)

app = graph.compile()

if __name__ == "__main__":
    intput_state = {
        "source": "The qualities that determine a subculture as distinct may be linguistic, aesthetic, religious, political, sexual, geographical, or a combination of factors.",
        "mt": "उपसंस्कृति को विशिष्ट रूप से निर्धारित करने वाले गुण भाषाई, सौंदर्य, धार्मिक, राजनीतिक, यौन, भौगोलिक या कारकों का संयोजन हो सकते हैं।",
        "reference": "वे गुण जो किसी उप-संस्कृति को अलग बनाते हैं, जैसे कि भाषा, सौंदर्य, धर्म, राजनीति, यौन, भूगोल या कई सारे कारकों का मिश्रण हो सकते हैं."
    }

    result = app.invoke(intput_state)
    print(result)
