import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

# ตั้งค่า LLM ที่ใช้
# คุณต้องมี Hugging Face Hub API Token
# โดยไปที่ https://huggingface.co/settings/tokens
# และตั้งค่าใน secrets ของ Streamlit หรือในโค้ด
# st.secrets["HUGGINGFACEHUB_API_TOKEN"] = "hf_..."
HUGGINGFACEHUB_API_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")

# ตรวจสอบว่ามี API token หรือไม่
if not HUGGINGFACEHUB_API_TOKEN:
    st.error("กรุณาตั้งค่า HUGGINGFACEHUB_API_TOKEN ใน Streamlit Secrets.")
    st.stop()

# โหลดโมเดล LLM จาก Hugging Face Hub
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",  # คุณสามารถเปลี่ยนเป็นโมเดลอื่นได้
    model_kwargs={"temperature": 0.5, "max_length": 512},
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)

# ฟังก์ชันสำหรับประมวลผลข้อมูลและสร้าง Vector Store
def get_vector_store():
    # โหลดข้อมูลจากไฟล์ data.txt
    try:
        with open("data.txt", "r", encoding="utf-8") as file:
            raw_text = file.read()
    except FileNotFoundError:
        st.error("ไม่พบไฟล์ data.txt โปรดสร้างไฟล์นี้และใส่ข้อมูลลงไป")
        return None

    # แบ่งข้อความเป็นส่วนย่อย (chunks)
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,  # ขนาดของแต่ละส่วน
        chunk_overlap=200, # ขนาดส่วนที่ทับซ้อนกัน
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    # สร้าง Embeddings โดยใช้โมเดลจาก Hugging Face
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # สร้าง Vector Store และฝังข้อมูล
    vector_store = FAISS.from_texts(texts, embeddings)
    return vector_store

# สร้างหน้าเว็บแอปพลิเคชันด้วย Streamlit
st.set_page_config(page_title="ระบบตอบคำถามจากข้อมูล", page_icon="🤖")

st.header("🤖 ระบบตอบคำถามจากข้อมูลส่วนตัว")
st.write("ป้อนคำถามเกี่ยวกับข้อมูลในไฟล์ data.txt เพื่อรับคำตอบ")

# โหลด Vector Store เมื่อเริ่มแอปครั้งแรก
if "vector_store" not in st.session_state:
    with st.spinner("กำลังประมวลผลข้อมูล..."):
        st.session_state.vector_store = get_vector_store()
    if st.session_state.vector_store is None:
        st.stop()

# สร้าง Chain สำหรับการตอบคำถามแบบ RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=st.session_state.vector_store.as_retriever(),
)

# สร้าง input field ให้ผู้ใช้ป้อนคำถาม
query = st.text_input("ป้อนคำถามของคุณที่นี่:", key="query_input")

if query:
    if st.session_state.vector_store is None:
        st.warning("ไม่สามารถประมวลผลคำถามได้ เนื่องจาก Vector Store ไม่พร้อม")
    else:
        # แสดงผลลัพธ์การตอบคำถาม
        with st.spinner("กำลังค้นหาและสร้างคำตอบ..."):
            response = qa_chain.run(query)
            st.success("นี่คือคำตอบ:")
            st.info(response)
