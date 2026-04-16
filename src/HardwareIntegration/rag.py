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
If the vehicle has stopped (velocity is zero) and the stop sign is still far away (distance greater than 2), then take no action.
If the vehicle is close to the stop sign (distance less than or equal to 25) and is still moving faster than 2.5, then initiate braking (actuate).
If the vehicle is at a medium distance from the stop sign (between 50 and 60) and its speed is outside the safe range of 8 to 10, then issue a warning to adjust speed.
If the stop sign is far away (distance greater than 99) and the vehicle speed is outside the range of 8 to 12, then issue a warning indicating a stop sign ahead.
If the vehicle is very close to the stop sign (distance less than or equal to 2) and is already almost stopped (velocity less than or equal to 0.5), then take no action.
In all other situations, take no action.

YOU MUST:
- Follow numeric rules exactly
- Do not approximate conditions
- Do not override rules with language reasoning

STOP BEHAVIOR (s <= 2 and v <= 0.5):
If the head is centered, instruct the driver to turn their head to both the left and right sides.
If the eyes are centered, instruct the driver to check both sides using their eyes.
If both the head and eyes are centered, instruct the driver to perform both actions (turn the head and check with the eyes).
If the driver is already looking toward one side, guide them to check the opposite side.

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
        options={"temperature": 0.0, "num_predict": 20},
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