from pathlib import Path
import json
import os
import re

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

load_dotenv()

DATA_DIR = Path("./")

app = FastAPI(title="Agentic AI Support Triage Assistant")

DOCS = {}
TICKETS = []
ACCOUNTS = []
VECTORSTORE = None

VALID_ROUTES = {
    "KNOWLEDGE_BASE",
    "TICKET_LOOKUP",
    "ACCOUNT_LOOKUP",
    "AMBIGUOUS",
    "UNSUPPORTED",
}


class QueryRequest(BaseModel):
    question: str


def load_markdown_docs(data_dir: Path) -> dict[str, str]:
    docs = {}
    for md_file in sorted(data_dir.glob("*.md")):
        docs[md_file.name] = md_file.read_text(encoding="utf-8").strip()
    return docs


def load_tickets(data_dir: Path) -> list[dict]:
    tickets_path = data_dir / "tickets.json"
    with open(tickets_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_accounts(data_dir: Path) -> list[dict]:
    accounts_path = data_dir / "accounts.json"
    with open(accounts_path, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_markdown_docs(docs: dict[str, str]) -> list[dict]:
    chunks = []
    for source, full_text in docs.items():
        parts = [part.strip() for part in full_text.split("\n\n") if part.strip()]
        for idx, part in enumerate(parts):
            chunks.append(
                {
                    "source": source,
                    "chunk_id": f"{source}::{idx}",
                    "text": part,
                }
            )
    return chunks


def build_vectorstore(chunks: list[dict]) -> FAISS:
    documents = [
        Document(
            page_content=chunk["text"],
            metadata={
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
            },
        )
        for chunk in chunks
    ]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(documents, embeddings)


def get_customer_names(accounts: list[dict]) -> list[str]:
    return [a["customer_name"].strip().lower() for a in accounts]


def llm_route_query(question: str) -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""
You are a query router for a SaaS support triage assistant.

Classify the user query into exactly one of these routes:
- KNOWLEDGE_BASE
- TICKET_LOOKUP
- ACCOUNT_LOOKUP
- AMBIGUOUS
- UNSUPPORTED

Rules:
- KNOWLEDGE_BASE: policy, process, setup, rate limit, refund, upgrade, security, integration questions answered from markdown docs
- TICKET_LOOKUP: asks about a ticket, ticket ID, ticket status, assignee, open/unassigned/urgent tickets
- ACCOUNT_LOOKUP: asks about customer account details like plan, renewal date, health score, open ticket count
- AMBIGUOUS: unclear target, missing context, should ask clarifying question
- UNSUPPORTED: clear question, but information is not available in the provided sources

Examples:
User query: What is your refund policy?
Output: {{"route":"KNOWLEDGE_BASE","confidence":0.95}}

User query: Who is assigned to T-2003?
Output: {{"route":"TICKET_LOOKUP","confidence":0.99}}

User query: Which urgent tickets are still open?
Output: {{"route":"TICKET_LOOKUP","confidence":0.96}}

User query: When does Acme Corp renew?
Output: {{"route":"ACCOUNT_LOOKUP","confidence":0.97}}

User query: What plan is Delta Retail on?
Output: {{"route":"ACCOUNT_LOOKUP","confidence":0.96}}

User query: Check that ticket for me
Output: {{"route":"AMBIGUOUS","confidence":0.93}}

User query: What is going on with Acme?
Output: {{"route":"AMBIGUOUS","confidence":0.90}}

User query: Can you look at the integration issue?
Output: {{"route":"AMBIGUOUS","confidence":0.88}}

User query: Do you support on-premise deployment?
Output: {{"route":"UNSUPPORTED","confidence":0.94}}

User query: What are the legal policies for Germany?
Output: {{"route":"UNSUPPORTED","confidence":0.92}}

Return only valid JSON in this format:
{{"route":"<ONE_ROUTE>","confidence":0.0}}

User query: {question}
"""
    response = llm.invoke(prompt)
    text = response.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(text)
        route = data.get("route", "").strip()
        confidence = float(data.get("confidence", 0.5))

        if route not in VALID_ROUTES:
            return {"route": "AMBIGUOUS", "confidence": 0.5}

        return {"route": route, "confidence": confidence}
    except Exception:
        return {"route": "AMBIGUOUS", "confidence": 0.5}


def route_query(question: str, accounts: list[dict]) -> dict:
    q = question.strip().lower()
    customer_names = get_customer_names(accounts)

    ambiguous_phrases = [
        "that ticket",
        "check that ticket",
        "what is going on with",
        "look at the integration issue",
        "can you look at the integration issue",
    ]
    if any(p in q for p in ambiguous_phrases):
        return {"route": "AMBIGUOUS", "confidence": 0.8}

    if re.search(r"\bT-\d+\b", question, re.IGNORECASE):
        return {"route": "TICKET_LOOKUP", "confidence": 0.98}
    ticket_keywords = [
        "ticket",
        "assigned",
        "unassigned",
        "urgent tickets",
        "status of ticket"
    ]
    if any(k in q for k in ticket_keywords):
        return {"route": "TICKET_LOOKUP", "confidence": 0.9}

    account_keywords = [
        "plan", "renewal", "renew", "health score",
        "open ticket count", "account", "open tickets",
        "low health", "accounts", "customers"
    ]

    if any(k in q for k in ["low health", "health score", "open ticket count", "open tickets", "accounts"]):
        return {"route": "ACCOUNT_LOOKUP", "confidence": 0.92}

    if any(name in q for name in customer_names) and any(k in q for k in account_keywords):
        return {"route": "ACCOUNT_LOOKUP", "confidence": 0.92}

    kb_keywords = [
        "refund",
        "upgrade",
        "downgrade",
        "rate limit",
        "429",
        "security",
        "integration",
        "webhook",
        "api key",
    ]
    if any(k in q for k in kb_keywords):
        return {"route": "KNOWLEDGE_BASE", "confidence": 0.88}

    return llm_route_query(question)


def find_account_by_name(accounts: list[dict], question: str):
    q = question.lower()
    for account in accounts:
        if account["customer_name"].lower() in q:
            return account
    return None


def handle_ticket_lookup(question: str, tickets: list[dict]) -> dict:
    q = question.lower()

    ticket_id_match = re.search(r"\bT-\d+\b", question, re.IGNORECASE)
    if ticket_id_match:
        ticket_id = ticket_id_match.group(0).upper()
        ticket = next((t for t in tickets if t["ticket_id"].upper() == ticket_id), None)

        if not ticket:
            return {
                "route": "UNSUPPORTED",
                "confidence": 0.95,
                "used_sources": ["tickets.json"],
                "used_tools": ["ticket_lookup"],
                "needs_clarification": False,
                "final_answer": f"I could not find ticket {ticket_id} in tickets.json.",
            }

        if "assigned" in q:
            assignee = ticket["assigned_to"]
            if assignee in [None, "", "None"]:
                answer = f"Ticket {ticket_id} is currently unassigned."
            else:
                answer = f"Ticket {ticket_id} is assigned to {assignee}."
        elif "status" in q:
            answer = f"Ticket {ticket_id} is currently {ticket['status']}."
        else:
            assignee = ticket["assigned_to"] if ticket["assigned_to"] not in [None, "", "None"] else "unassigned"
            answer = (
                f"Ticket {ticket_id} is {ticket['status']}, assigned to {assignee}, "
                f"with priority {ticket['priority']} for {ticket['customer_name']}."
            )

        return {
            "route": "TICKET_LOOKUP",
            "confidence": 0.98,
            "used_sources": ["tickets.json"],
            "used_tools": ["ticket_lookup"],
            "needs_clarification": False,
            "final_answer": answer,
        }

    if "urgent" in q and "open" in q:
        matches = [t for t in tickets if t["priority"].lower() == "urgent" and t["status"].lower() == "open"]
        if not matches:
            return {
                "route": "UNSUPPORTED",
                "confidence": 0.9,
                "used_sources": ["tickets.json"],
                "used_tools": ["ticket_lookup"],
                "needs_clarification": False,
                "final_answer": "I could not find any urgent tickets that are still open.",
            }

        answer = "Urgent open tickets: " + "; ".join(
            [
                f"{t['ticket_id']} ({t['customer_name']}) assigned to {t['assigned_to'] if t['assigned_to'] not in [None, '', 'None'] else 'unassigned'}"
                for t in matches
            ]
        )
        return {
            "route": "TICKET_LOOKUP",
            "confidence": 0.94,
            "used_sources": ["tickets.json"],
            "used_tools": ["ticket_lookup"],
            "needs_clarification": False,
            "final_answer": answer,
        }

    if "unassigned" in q:
        matches = [
            t
            for t in tickets
            if not t["assigned_to"] or str(t["assigned_to"]).strip().lower() in ["unassigned", "none", "null", ""]
        ]
        if not matches:
            return {
                "route": "UNSUPPORTED",
                "confidence": 0.9,
                "used_sources": ["tickets.json"],
                "used_tools": ["ticket_lookup"],
                "needs_clarification": False,
                "final_answer": "I could not find any currently unassigned tickets.",
            }

        answer = "Unassigned tickets: " + "; ".join([f"{t['ticket_id']} ({t['customer_name']})" for t in matches])
        return {
            "route": "TICKET_LOOKUP",
            "confidence": 0.93,
            "used_sources": ["tickets.json"],
            "used_tools": ["ticket_lookup"],
            "needs_clarification": False,
            "final_answer": answer,
        }

    return {
        "route": "AMBIGUOUS",
        "confidence": 0.75,
        "used_sources": ["tickets.json"],
        "used_tools": ["ticket_lookup"],
        "needs_clarification": True,
        "final_answer": "Please provide a ticket ID or specify whether you want status, assignee, urgent open tickets, or unassigned tickets.",
    }


def handle_account_lookup(question: str, accounts: list[dict]) -> dict:
    q = question.lower()
    account = find_account_by_name(accounts, question)

    if "low health" in q and "open tickets" in q:
        matches = [a for a in accounts if a["health_score"] < 50 and a["open_ticket_count"] > 0]
        if not matches:
            return {
                "route": "UNSUPPORTED",
                "confidence": 0.9,
                "used_sources": ["accounts.json"],
                "used_tools": ["account_lookup"],
                "needs_clarification": False,
                "final_answer": "I could not find any customers with open tickets and a low health score.",
            }

        answer = "Customers with open tickets and low health scores: " + "; ".join(
            [
                f"{a['customer_name']} (health score {a['health_score']}, open tickets {a['open_ticket_count']})"
                for a in matches
            ]
        )
        return {
            "route": "ACCOUNT_LOOKUP",
            "confidence": 0.94,
            "used_sources": ["accounts.json"],
            "used_tools": ["account_lookup"],
            "needs_clarification": False,
            "final_answer": answer,
        }

    if "low health" in q:
        matches = [a for a in accounts if a["health_score"] < 50]
        if not matches:
            return {
                "route": "UNSUPPORTED",
                "confidence": 0.9,
                "used_sources": ["accounts.json"],
                "used_tools": ["account_lookup"],
                "needs_clarification": False,
                "final_answer": "I could not find any accounts with low health scores.",
            }

        answer = "Accounts with low health scores: " + "; ".join(
            [f"{a['customer_name']} ({a['health_score']})" for a in matches]
        )
        return {
            "route": "ACCOUNT_LOOKUP",
            "confidence": 0.93,
            "used_sources": ["accounts.json"],
            "used_tools": ["account_lookup"],
            "needs_clarification": False,
            "final_answer": answer,
        }

    if account:
        if "plan" in q:
            answer = f"{account['customer_name']} is on the {account['plan']} plan."
        elif "renew" in q:
            answer = f"{account['customer_name']} renews on {account['renewal_date']}."
        elif "health" in q:
            answer = f"{account['customer_name']} has a health score of {account['health_score']}."
        elif "open ticket" in q:
            answer = f"{account['customer_name']} has {account['open_ticket_count']} open tickets."
        else:
            answer = (
                f"{account['customer_name']} is on the {account['plan']} plan, renews on "
                f"{account['renewal_date']}, has {account['open_ticket_count']} open tickets, "
                f"and a health score of {account['health_score']}."
            )

        return {
            "route": "ACCOUNT_LOOKUP",
            "confidence": 0.96,
            "used_sources": ["accounts.json"],
            "used_tools": ["account_lookup"],
            "needs_clarification": False,
            "final_answer": answer,
        }

    return {
        "route": "UNSUPPORTED",
        "confidence": 0.9,
        "used_sources": ["accounts.json"],
        "used_tools": ["account_lookup"],
        "needs_clarification": False,
        "final_answer": "I could not find that customer account in accounts.json.",
    }


def handle_knowledge_base(question: str, vectorstore: FAISS) -> dict:
    results = vectorstore.similarity_search(question, k=3)

    if not results:
        return {
            "route": "UNSUPPORTED",
            "confidence": 0.85,
            "used_sources": [],
            "used_tools": ["kb_retrieval"],
            "needs_clarification": False,
            "final_answer": "I could not find relevant information in the knowledge base.",
        }

    context = "\n\n".join([doc.page_content for doc in results])
    used_sources = sorted(list({doc.metadata["source"] for doc in results}))

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""
You are answering a SaaS support question using only the provided context.
Do not invent anything.
If the answer is not clearly supported by the context, say that the information is unavailable in the provided knowledge base.

Question:
{question}

Context:
{context}

Return only the final answer text.
"""
    response = llm.invoke(prompt)
    final_answer = response.content.strip()

    if "unavailable" in final_answer.lower() or "not available" in final_answer.lower():
        return {
            "route": "UNSUPPORTED",
            "confidence": 0.85,
            "used_sources": used_sources,
            "used_tools": ["kb_retrieval"],
            "needs_clarification": False,
            "final_answer": final_answer,
        }

    return {
        "route": "KNOWLEDGE_BASE",
        "confidence": 0.9,
        "used_sources": used_sources,
        "used_tools": ["kb_retrieval"],
        "needs_clarification": False,
        "final_answer": final_answer,
    }


def handle_ambiguous() -> dict:
    return {
        "route": "AMBIGUOUS",
        "confidence": 0.8,
        "used_sources": [],
        "used_tools": [],
        "needs_clarification": True,
        "final_answer": "Your request is ambiguous. Please provide a ticket ID, customer name, or a more specific question.",
    }


def handle_unsupported() -> dict:
    return {
        "route": "UNSUPPORTED",
        "confidence": 0.9,
        "used_sources": [],
        "used_tools": [],
        "needs_clarification": False,
        "final_answer": "The requested information is not available in the provided knowledge base or structured data.",
    }


def dispatch_query(question: str, accounts: list[dict], tickets: list[dict], vectorstore: FAISS) -> dict:
    route_result = route_query(question, accounts)
    route = route_result["route"]

    if route == "KNOWLEDGE_BASE":
        return handle_knowledge_base(question, vectorstore)
    if route == "TICKET_LOOKUP":
        return handle_ticket_lookup(question, tickets)
    if route == "ACCOUNT_LOOKUP":
        return handle_account_lookup(question, accounts)
    if route == "AMBIGUOUS":
        return handle_ambiguous()
    return handle_unsupported()


@app.on_event("startup")
def startup_event():
    global DOCS, TICKETS, ACCOUNTS, VECTORSTORE

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is missing in environment.")

    DOCS = load_markdown_docs(DATA_DIR)
    TICKETS = load_tickets(DATA_DIR)
    ACCOUNTS = load_accounts(DATA_DIR)
    chunks = chunk_markdown_docs(DOCS)
    VECTORSTORE = build_vectorstore(chunks)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query(request: QueryRequest):
    return dispatch_query(
        question=request.question,
        accounts=ACCOUNTS,
        tickets=TICKETS,
        vectorstore=VECTORSTORE,
    )