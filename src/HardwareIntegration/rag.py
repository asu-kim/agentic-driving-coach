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

Rules for TOKEN:
- If s <= 25 and v > 2.5 → ACTUATE
- If 50 <= s <= 60 and v not in [8,10] → WARNING
- If s > 99 and v not in [8,12] → WARNING
- Otherwise → NONE

Rules for Message:
- One short sentence only
- Friendly driving coach tone
- Mention stop sign behavior
- Guide driver using head/eye/steer if relevant
- If TOKEN=NONE → still give light guidance or appreciation


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


def generate_response(s, v, st, h, e):
    query = (
        f"s={s:.2f}, v={v:.2f}, st={st}, h={h}, e={e}. "
        "Give driving guidance for approaching a stop sign."
    )

    docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
            {context}

            Query: {query}

            Output exactly ONE line in format TOKEN|Message
            """

    try:
        raw = llm(prompt)

        # Safe parsing
        if "|" in raw:
            token, msg = raw.split("|", 1)
            return token.strip(), msg.strip()
        else:
            return "NONE", ""

    except Exception as e:
        print("RAG ERROR:", e, flush=True)
        return "NONE", ""


if __name__ == "__main__":
    s = 20.0
    v = 5.0
    st, h, e = 2, 2, 2

    token, msg = generate_response(s, v, st, h, e)
    print("OUTPUT:", token, "|", msg)