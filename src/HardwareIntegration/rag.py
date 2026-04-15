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
- LEFT = 0 -> looking/steering left
- CENTER = 1 -> straight
- RIGHT = 2 -> looking/steering right

Rules for TOKEN:
- If s <= 25 and v > 2.5 -> ACTUATE
- If 50 <= s <= 60 and v not in [8,10] -> WARNING
- If s > 99 and v not in [8,12] -> WARNING
- If s <= 2 and v <= 0.5 -> ACTUATE
- Otherwise -> NONE

IF S == 0.5 AND V == 0.5:
CRITICAL BEHAVIOR RULES:
- You MUST use h and e to determine driver awareness
- If h == 1 (CENTER) -> driver is NOT checking sides -> instruct head movement
- If e == 1 (CENTER) -> driver is NOT scanning -> instruct eye movement
- If both h and e are CENTER at stop ->  "turn your head left and right and check both sides"
- If h or e already LEFT/RIGHT -> acknowledge and guide next action

Rules for Message:
- One short sentence only
- MUST include action based on current values
- DO NOT give generic advice
- DO NOT repeat the same message
- MUST reference behavior (head/eye/steer) when s <= 10

If TOKEN=NONE -> still give light but specific feedback based on inputs
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
            {"role": "system", "content": "You are a calm and helpful driving coach."},
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0.0, "num_predict": 30},
    )
    return (response.get("message", {}).get("content", "") or "").strip()


def generate_response(self, s, v, st, h, e):
    query = f"""
            Current driving state:
            - Distance to stop (s): {s:.2f} meters
            - Velocity (v): {v:.2f} m/s
            - Steering (st): {st}
            - Head position (h): {h}
            - Eye position (e): {e}

            Interpret and generate driving guidance. IF STOP IS REACHED AND S IS 0 CHECK LEFT OR RIGHT
            """

    docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
            {context}

            You MUST base your response on ALL provided variables (s, v, st, h, e) and be natural.

            Query:
            {query}

            Output exactly ONE line in format TOKEN|Message
            """

    try:
        raw = llm(prompt)

      
        if "|" in raw:
            token, msg = raw.split("|", 1)
            return token.strip(), msg.strip()
        else:
            return "NONE", ""

    except Exception as e:
        print("RAG ERROR:", e, flush=True)
        return "NONE", ""
