from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from typing import TypedDict, Dict, List, Optional, Literal, Annotated, Any
from own_framework_prompts import *
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

class AgentOutputStage1(BaseModel):
    probability: float = Field(..., description="The probability that the error is present", ge=0.0, le=1.0)
    reason: str = Field(..., description="Explanation pointing to concrete words, phrases, or structures that justify the probability.")
    confidence: float = Field(..., description="Score out of 100 on how confident you are that this error is present in the sentence.", ge=0.0, le=100.0)

class AgentOutputStage2(BaseModel):
    reEvaluatedProb: float = Field(..., description="Based on your evaluation re evaluate the probability while considering the probability given by the previous agent.", ge=0.0, le=1.0)
    thoughtsOnStage1: str = Field(..., description="Based on your evaluations, what are your thoughts on evaluations of the previous agent?")
    reason: str = Field(..., description="Give a brief explanation of your thoughts on the previous agent's evaluations. If you agree of disagree with the previous evaluation give concrete evidence for your specific error.")
    reEvaluatedConfidence: float = Field(..., description="Based on your evaluations score out of 100 on how confident are that this error is present in the sentence and that your thoughts and explanations are valid", ge=0.0, le=100.0)

class AgentOutputStage3(BaseModel):
    consistencyScore: float = Field(..., description="Based on the evaluations of the previous agents, generate a score out of 100 on how consistent the agents are with each other.")
    errorsExists: str = Field(..., description="Based on the evaluations of the previous agents i want you to verify whether these errors exists or not. Dont re evaluate. You have to search whether the error flagged by the previous agents exists or not. If it exists then return 'YES' otherwise return 'NO' and mention which errors for both")
    existanceReasoning: str = Field(..., description="Give brief explanation on your verification of the existance of the errors.")

class MTState(TypedDict):
    source: str
    mt: str
    reference: str

    accuracyStage1: Optional[AgentOutputStage1]
    fluencyStage1: Optional[AgentOutputStage1]
    terminologyStage1: Optional[AgentOutputStage1]
    styleStage1: Optional[AgentOutputStage1]
    
    addition: Optional[AgentOutputStage2]
    omission: Optional[AgentOutputStage2]
    mistranslation: Optional[AgentOutputStage2]
    untranslated_text: Optional[AgentOutputStage2]
    punctuation: Optional[AgentOutputStage2]
    spelling: Optional[AgentOutputStage2]
    grammar: Optional[AgentOutputStage2]
    register: Optional[AgentOutputStage2]
    inconsistency: Optional[AgentOutputStage2]
    characterEncoding: Optional[AgentOutputStage2]
    inappropriate_for_context: Optional[AgentOutputStage2]
    inconsistency_use: Optional[AgentOutputStage2]
    awkward: Optional[AgentOutputStage2]

    accuracyStage3: Optional[AgentOutputStage3]
    fluencyStage3: Optional[AgentOutputStage3]
    terminologyStage3: Optional[AgentOutputStage3]
    styleStage3: Optional[AgentOutputStage3]


llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

def make_error_agent_stage1(system_prompt: str, state_key: str):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """
        SOURCE SENTENCE: {source}
         
        MACHINE TRANSLATED SENTENCE: {translated}
         
        REFERENCE SENTENCE: {reference}
        """)
    ])

    chain = prompt_template | llm.with_structured_output(AgentOutputStage1)

    def agent_fn(state: MTState) -> Dict[str, AgentOutputStage1]:
        output = chain.invoke({
            "source": state["source"],
            "translated": state["mt"],
            "reference": state["reference"],
        })

        return {state_key: output}
    return agent_fn

def make_error_agent_stage2(system_prompt: str, state_key: str, SuperCategory: str):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """
        SOURCE SENTENCE: {source}
         
        MACHINE TRANSLATED SENTENCE: {translated}
         
        REFERENCE SENTENCE: {reference}
        
        PREVIOUS AGENT EVALUATIONS: {previous_agent}
        """)
    ])

    chain = prompt_template | llm.with_structured_output(AgentOutputStage2)

    def agent_fn_stage2(state: MTState) -> Dict[str, AgentOutputStage2]:
        output = chain.invoke({
            "source": state["source"],
            "translated": state["mt"],
            "reference": state["reference"],
            "previous_agent": state[SuperCategory]
        })

        return {state_key: output}
    return agent_fn_stage2

def make_error_agent_stage3(system_prompt: str, state_key: str, SuperCategory: str):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """
        SOURCE SENTENCE: {source}
         
        MACHINE TRANSLATED SENTENCE: {translated}
         
        REFERENCE SENTENCE: {reference}
        
        SUPER CATEGORY AGENT EVALUATIONS: {previous_agent}
         
        SUB CATEGORY AGENTS EVALUATIONS: {sub_category_agent}
        """)
    ])

    chain = prompt_template | llm.with_structured_output(AgentOutputStage3)

    def agent_fn_stage3(state: MTState) -> Dict[str, AgentOutputStage3]:
        if SuperCategory == "accuracyStage1":
            sub = ["addition", "omission", "mistranslation", "untranslated_text"]
        elif SuperCategory == "fluencyStage1":
            sub = ["punctuation", "spelling", "grammar", "register", "inconsistency", "characterEncoding"]
        elif SuperCategory == "terminologyStage1":
            sub = ["inappropriate_for_context", "inconsistency_use"]
        elif SuperCategory == "styleStage1":
            sub = ["awkward"]
        
        combined_sub_category = [state[s] for s in sub if state.get(s) is not None]

        output = chain.invoke({
            "source": state["source"],
            "translated": state["mt"],
            "reference": state["reference"],
            "previous_agent": state[SuperCategory],
            "sub_category_agent": combined_sub_category
        })

        return {state_key: output}
    return agent_fn_stage3


accuracy_agent = make_error_agent_stage1(ACCURACY_PROMPT, "accuracyStage1")
fluency_agent = make_error_agent_stage1(FLUENCY_PROMPT, "fluencyStage1")
terminology_agent = make_error_agent_stage1(TERMINOLOGY_PROMPT, "terminologyStage1")
style_agent = make_error_agent_stage1(STYLE_PROMPT, "styleStage1")

addition_agent = make_error_agent_stage2(ADDITION_PROMPT, "addition", "accuracyStage1")
omission_agent = make_error_agent_stage2(OMISSION_PROMPT, "omission", "accuracyStage1")
mistranslation_agent = make_error_agent_stage2(MISTRANSLATION_PROMPT, "mistranslation", "accuracyStage1")
untranslated_text_agent = make_error_agent_stage2(UNTRANSLATED_TEXT_PROMPT, "untranslated_text", "accuracyStage1")
punctuation_agent = make_error_agent_stage2(PUNCTUATION_PROMPT, "punctuation", "fluencyStage1")
spelling_agent = make_error_agent_stage2(SPELLING_PROMPT, "spelling", "fluencyStage1")
grammar_agent = make_error_agent_stage2(GRAMMAR_PROMPT, "grammar", "fluencyStage1")
register_agent = make_error_agent_stage2(REGISTER_PROMPT, "register", "fluencyStage1")
inconsistency_agent = make_error_agent_stage2(INCONSISTENCY_PROMPT, "inconsistency", "fluencyStage1")
characterEncoding_agent = make_error_agent_stage2(CHARACTER_ENCODING_PROMPT, "characterEncoding", "fluencyStage1")
inappropriate_for_context_agent = make_error_agent_stage2(INAPPROPRIATE_FOR_CONTEXT_PROMPT, "inappropriate_for_context", "terminologyStage1")
inconsistency_use_agent = make_error_agent_stage2(INCONSISTENT_USE_PROMPT, "inconsistency_use", "terminologyStage1")
awkward_agent = make_error_agent_stage2(AWKWARD_PROMPT, "awkward", "styleStage1")

accuracy_stage3_agent = make_error_agent_stage3(ACCURACY_STAGE3_PROMPT, "accuracyStage3", "accuracyStage1")
fluency_stage3_agent = make_error_agent_stage3(FLUENCY_STAGE3_PROMPT, "fluencyStage3", "fluencyStage1")
terminology_stage3_agent = make_error_agent_stage3(TERMINOLOGY_STAGE3_PROMPT, "terminologyStage3", "terminologyStage1")
style_stage3_agent = make_error_agent_stage3(STYLE_STAGE3_PROMPT, "styleStage3", "styleStage1")

graph = StateGraph(MTState)

graph.add_node("START")
graph.add_node("accuracyStage1", accuracy_agent)
graph.add_node("fluencyStage1", fluency_agent)
graph.add_node("terminologyStage1", terminology_agent)
graph.add_node("styleStage1", style_agent)
graph.add_node("addition", addition_agent)
graph.add_node("omission", omission_agent)
graph.add_node("mistranslation", mistranslation_agent)
graph.add_node("untranslated_text", untranslated_text_agent)
graph.add_node("punctuation", punctuation_agent)
graph.add_node("spelling", spelling_agent)
graph.add_node("grammar", grammar_agent)
graph.add_node("register", register_agent)
graph.add_node("inconsistency", inconsistency_agent)
graph.add_node("characterEncoding", characterEncoding_agent)
graph.add_node("inappropriate_for_context", inappropriate_for_context_agent)
graph.add_node("inconsistency_use", inconsistency_use_agent)
graph.add_node("awkward", awkward_agent)
graph.add_node("accuracyStage3", accuracy_stage3_agent)
graph.add_node("fluencyStage3", fluency_stage3_agent)
graph.add_node("terminologyStage3", terminology_stage3_agent)
graph.add_node("styleStage3", style_stage3_agent)

graph.set_entry_point("START")

graph.add_edge("START", "accuracyStage1")
graph.add_edge("START", "fluencyStage1")
graph.add_edge("START", "terminologyStage1")
graph.add_edge("START", "styleStage1")

graph.add_edge("accuracyStage1", "addition")
graph.add_edge("accuracyStage1", "omission")
graph.add_edge("accuracyStage1", "mistranslation")
graph.add_edge("accuracyStage1", "untranslated_text")

graph.add_edge("fluencyStage1", "punctuation")
graph.add_edge("fluencyStage1", "spelling")
graph.add_edge("fluencyStage1", "grammar")
graph.add_edge("fluencyStage1", "register")
graph.add_edge("fluencyStage1", "inconsistency")
graph.add_edge("fluencyStage1", "characterEncoding")

graph.add_edge("terminologyStage1", "inappropriate_for_context")
graph.add_edge("terminologyStage1", "inconsistency_use")

graph.add_edge("styleStage1", "awkward")

graph.add_edge("addition", "accuracyStage3")
graph.add_edge("omission", "accuracyStage3")
graph.add_edge("mistranslation", "accuracyStage3")
graph.add_edge("untranslated_text", "accuracyStage3")

graph.add_edge("punctuation", "fluencyStage3")
graph.add_edge("spelling", "fluencyStage3")
graph.add_edge("grammar", "fluencyStage3")
graph.add_edge("register", "fluencyStage3")
graph.add_edge("inconsistency", "fluencyStage3")
graph.add_edge("characterEncoding", "fluencyStage3")

graph.add_edge("inappropriate_for_context", "terminologyStage3")
graph.add_edge("inconsistency_use", "terminologyStage3")

graph.add_edge("awkward", "styleStage3")