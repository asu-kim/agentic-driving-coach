import csv
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

KG = []
with open("kg_rules.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        KG.append(row)

def evaluate_condition(cond, s, v, st, h, e):
    try:
        cond = cond.replace("&", "and")
        return eval(cond)
    except:
        return False

def get_token_from_kg(s, v, st, h, e):
    for row in KG:
        if row["type"] != "rule":
            continue
        if evaluate_condition(row["subject"], s, v, st, h, e):
            return row["object"]
    return "NONE"

Rules = """Output exactly ONE line: TOKEN|Message

        Variables:
        - s = distance to stop
        - v = velocity
        - st = steering
        - h = head
        - e = eye

        Use h and e strictly for guidance.
        Generate only one short sentence.
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
            {"role": "system", "content": "You are a driving coach."},
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0.0, "num_predict": 30},
    )
    return (response.get("message", {}).get("content", "") or "").strip()

def generate_response(s, v, st, h, e):
    token = get_token_from_kg(s, v, st, h, e)

    query = f"""
    s={s}, v={v}, st={st}, h={h}, e={e}
    """

    docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
    {context}

    TOKEN is fixed: {token}

    Generate message using h and e.

    Output: {token}|Message
    """

    raw = llm(prompt)

    if "|" in raw:
        _, msg = raw.split("|", 1)
        return token, msg.strip()
    else:
        return token, ""

