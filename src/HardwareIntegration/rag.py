from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from torch import cuda
import ollama


device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'


embed_model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
)


Rules = """Output exactly ONE line: TOKEN|Message

TOKEN must be one of: NONE, WARNING, ACTUATE

Variables:
- s = distance to stop (meters)
- v = velocity (m/s)
- st = steering (0=LEFT,1=CENTER,2=RIGHT)
- h = head position (0=LEFT,1=CENTER,2=RIGHT)
- e = eye position (0=LEFT,1=CENTER,2=RIGHT)

Interpretation:
- LEFT = 0
- CENTER = 1
- RIGHT = 2

STRICT TOKEN RULES:
- If v == 0 and s > 2 -> NONE
- If s <= 25 and v > 2.5 -> ACTUATE
- If 50 <= s <= 60 and v not in [8,10] -> WARNING
- If s > 99 and v not in [8,12] -> WARNING
- If s <= 2 and v <= 0.5 -> NONE
- Otherwise -> NONE

YOU MUST:
- Follow numeric rules exactly
- Do not approximate conditions
- Do not override rules with language reasoning

STOP BEHAVIOR (s <= 2 and v <= 0.5):
- If h == 1 -> instruct turning head LEFT and RIGHT
- If e == 1 -> instruct checking both sides with eyes
- If both h and e == 1 -> instruct BOTH actions
- If already looking one side -> guide to check the other side

MESSAGE RULES:
- One short sentence only
- Must reference current state when s <= 10
- Must be action-specific
- No generic advice
- No repetition
"""


documents = [Document(page_content=Rules)]

split_docs = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
).split_documents(documents)

vectorstore = FAISS.from_documents(split_docs, embed_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})


def llm(prompt):
    response = ollama.chat(
        model="llama3:8b",
        messages=[
            {"role": "system", "content": "You strictly follow numeric rules and output only valid control tokens."},
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0.0, "num_predict": 30},
    )
    return (response.get("message", {}).get("content", "") or "").strip()


def generate_response(self, s, v, st, h, e):
    query = f"""
Current driving state:
- Distance to stop (s): {s:.2f}
- Velocity (v): {v:.2f}
- Steering (st): {st}
- Head position (h): {h}
- Eye position (e): {e}
"""

    docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
{context}

Step 1: Determine TOKEN strictly using numeric rules.
Step 2: Generate Message using state values.

Query:
{query}

Output exactly ONE line:
TOKEN|Message
"""

    try:
        raw = llm(prompt)

        if "|" in raw:
            token, msg = raw.split("|", 1)
            token = token.strip()
            msg = msg.strip()

            if token not in ["NONE", "WARNING", "ACTUATE"]:
                token = "NONE"

            return token, msg
        else:
            return "NONE", ""

    except Exception as e:
        print("RAG ERROR:", e, flush=True)
        return "NONE", ""