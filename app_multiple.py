import os
import json
import boto3
import streamlit as st
import logging
import base64
from dotenv import load_dotenv
from requests_aws4auth import AWS4Auth
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk
from unstructured.partition.pdf import partition_pdf
from io import BytesIO
from opensearchpy.exceptions import NotFoundError
from streamlit_pdf_viewer import pdf_viewer

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# ========================
# 1. 환경 변수 로드 및 클라이언트 설정
# ========================
load_dotenv()

aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION")
s3_bucket_name = os.getenv("S3_BUCKET_NAME")
opensearch_host = os.getenv("OPENSEARCH_COLLECTION_HOST")
opensearch_index_name = os.getenv("OPENSEARCH_INDEX_NAME")
bedrock_model_id = os.getenv("BEDROCK_MODEL_ID")

if not all([aws_access_key_id, aws_secret_access_key, aws_region, s3_bucket_name, opensearch_host, opensearch_index_name, bedrock_model_id]):
    st.error("`.env` 파일에 필요한 환경 변수가 설정되지 않았습니다. 모든 변수를 채워주세요.")
    st.stop()

session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)
credentials = session.get_credentials()
awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    aws_region,
    'aoss'
)

s3_client = session.client('s3')
bedrock_client = session.client('bedrock-runtime')
opensearch_client = OpenSearch(
    hosts=[{'host': opensearch_host, 'port': 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=30
)

# PDF 및 이미지 저장 폴더 설정 및 생성
PDF_DIR = "./pdf-files"
FIGURES_DIR = "./figures"
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ========================
# 앱 시작 시 PDF 파일 미리 다운로드
# ========================
pdf_file_name = "accu.pdf"
local_pdf_path = os.path.join(PDF_DIR, pdf_file_name)

if not os.path.exists(local_pdf_path) or os.path.getsize(local_pdf_path) == 0:
    st.info("PDF 파일을 S3에서 다운로드하고 있습니다. 잠시만 기다려주세요...")
    try:
        s3_client.download_file(s3_bucket_name, pdf_file_name, local_pdf_path)
        st.success("PDF 파일 다운로드 완료. 앱을 다시 시작합니다.")
        st.rerun()
    except Exception as e:
        st.error(f"S3에서 PDF 파일 다운로드 중 오류 발생: {e}")
        st.stop()


# ========================
# 2. 인덱싱 함수 (텍스트 + 이미지)
# ========================
def process_pdf_and_index_opensearch_with_images(s3_file_name):
    try:
        logging.info("S3에서 PDF 파일 다운로드 중...")
        s3_object = s3_client.get_object(Bucket=s3_bucket_name, Key=s3_file_name)
        pdf_content = s3_object['Body'].read()
        logging.info("다운로드 완료.")

        logging.info("unstructured를 사용하여 PDF 내용 분할 중...")
        elements = partition_pdf(
            file=BytesIO(pdf_content),
            strategy="hi_res",
            infer_table_structure=False,
            languages=["kor", "eng"],
            extract_images_in_pdf=True,
            image_output_dir_path="./figures"
        )
        logging.info(f"unstructured 분할 완료. 추출된 요소 수: {len(elements)}")

        pages = {}
        for el in elements:
            page_num = el.metadata.page_number if hasattr(el.metadata, 'page_number') else None
            if page_num not in pages:
                pages[page_num] = {'text': [], 'images': []}
            
            if el.text:
                pages[page_num]['text'].append(el.text)
            
            if hasattr(el.metadata, 'image_path') and el.metadata.image_path:
                pages[page_num]['images'].append(os.path.basename(el.metadata.image_path))

            
        print(f"OpenSearch 인덱스 '{opensearch_index_name}'이(가) 존재하지 않습니다. 새로 생성합니다.")
        
        mapping = {
            "settings": {
                "index.knn": True,
            },
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 1024,
                        "method": {
                            "name": "hnsw",
                            "engine": "nmslib"
                        }
                    },
                    "image_paths": {"type": "keyword"},
                    "page_number": {"type": "long"}
                }
            }
        }
                
        opensearch_client.indices.create(index=opensearch_index_name, body=mapping)
        logging.info(f"OpenSearch 인덱스 '{opensearch_index_name}' 생성 완료.")

        docs = []
        for page_num, content in pages.items():
            if not content['text'] and not content['images']:
                continue
            combined_text = "\n".join(content['text'])
            
            bedrock_model_id = 'cohere.embed-multilingual-v3'
            response = bedrock_client.invoke_model(
                modelId=bedrock_model_id,
                body=json.dumps({"texts": [combined_text], "input_type": "search_document"})
            )
            embedding = json.loads(response['body'].read())['embeddings'][0]

            doc = {
                'text': combined_text,
                'source': s3_file_name,
                'page_number': page_num,
                'embedding': embedding,
                'image_paths': content['images']
            }
            docs.append({
                '_index': opensearch_index_name,
                '_source': doc
            })

        # docs 리스트를 bulk로 한 번에 인덱싱
        if docs:
            try:
                successes, failures = bulk(opensearch_client, docs)
                if failures:
                    logging.error(f"인덱싱 실패: {failures}")
                else:
                    logging.info(f"PDF 내용 및 이미지 정보가 OpenSearch에 성공적으로 인덱싱되었습니다. 문서 수: {len(docs)}")
            except Exception as e:
                logging.error(f"벌크 인덱싱 중 오류 발생: {e}")
        else:
            logging.warning("인덱싱할 문서가 없습니다.")

        logging.info(f"PDF 내용 및 이미지 정보가 OpenSearch에 인덱싱되었습니다.")

    except Exception as e:
        logging.error(f"오류 발생: {e}")

# ========================
# 3. RAG 함수 (하이브리드 검색 + 이미지 출력)
# ========================
def get_rag_answer_from_bedrock_with_images(query):
    try:
        bedrock_model_id_embedding = 'cohere.embed-multilingual-v3'
        response = bedrock_client.invoke_model(
            modelId=bedrock_model_id_embedding,
            body=json.dumps({"texts": [query], "input_type": "search_query"})
        )
        query_embedding = json.loads(response['body'].read())['embeddings'][0]
        
        search_query = {
            "size": 3,
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["text^2"]
                            }
                        },
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_embedding,
                                    "k": 3
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "_source": ["text", "image_paths"]
        }
        
        response = opensearch_client.search(index=opensearch_index_name, body=search_query)
        hits = response['hits']['hits']
        context_docs = []
        image_paths = []
        
        for hit in hits:
            context_docs.append(hit['_source']['text'])
            if 'image_paths' in hit['_source'] and hit['_source']['image_paths']:
                image_paths.extend(hit['_source']['image_paths'])
        
        if not context_docs:
            return "죄송합니다. 질문에 대한 관련 문서를 찾을 수 없습니다.", []

        context = "\n\n".join(context_docs)
        
        llm_model_id = os.getenv("BEDROCK_MODEL_ID")
        
        if 'claude' in llm_model_id:
            prompt = f"""
            다음은 사용자 질문에 답변하기 위한 참고 자료입니다.
            <자료>
            {context}
            </자료>
            당신은 유용한 AI 비서입니다. 제공된 자료만을 바탕으로 사용자의 질문에 한국어로 상세하고 친절하게 답변해주세요. 만약 자료에 답변이 없다면, "죄송합니다. 제공된 문서에는 답변이 없습니다."라고 말해주세요. 절대 자료에 없는 정보를 임의로 생성하지 마세요.
            사용자 질문: {query}
            답변:
            """
            body = json.dumps({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 1000, "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}], "temperature": 0.5})
            response_body = bedrock_client.invoke_model(modelId=llm_model_id, body=body, contentType="application/json", accept="application/json")
            llm_answer = json.loads(response_body.get('body').read())['content'][0]['text']
        else:
            prompt = f"""
            You are a helpful AI assistant. Use only the provided context to answer the user's question in Korean. If the answer is not in the context, say "죄송합니다. 제공된 문서에는 답변이 없습니다." Do not make up information.
            Context:
            {context}
            Question: {query}
            Answer:
            """
            body = json.dumps({"inputText": prompt, "textGenerationConfig": {"maxTokenCount": 1000, "stopSequences": [], "temperature": 0.5, "topP": 0.9}})
            response_body = bedrock_client.invoke_model(modelId=llm_model_id, body=body)
            llm_answer = json.loads(response_body.get('body').read())['results'][0]['outputText']

        unique_image_paths = list(set(image_paths))
        return llm_answer, unique_image_paths

    except Exception as e:
        return f"RAG 처리 중 오류가 발생했습니다: {e}", []

# ========================
# 4. Streamlit 앱 로직
# ========================
st.set_page_config(layout="wide")
st.title("S3 기반 챗봇")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if 'text_input_value' not in st.session_state:
    st.session_state['text_input_value'] = ""

if 'pdf_page_number' not in st.session_state:
    st.session_state['pdf_page_number'] = 1

# PDF 뷰어와 챗봇 인터페이스를 두 개의 컬럼으로 나눕니다.
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("## PDF 미리보기 뷰어", unsafe_allow_html=True)
    pdf_path = os.path.join(PDF_DIR, "accu.pdf")
    if os.path.exists(pdf_path):
        pdf_file_data = open(pdf_path, "rb").read()
        
        # 페이지 넘기기 버튼 추가
        page_prev, page_info, page_next = st.columns([1, 1, 1])
        with page_prev:
            if st.button("이전 페이지"):
                if st.session_state.pdf_page_number > 1:
                    st.session_state.pdf_page_number -= 1
                    st.rerun()
        with page_info:
            st.markdown(f"**페이지: {st.session_state.pdf_page_number}**", unsafe_allow_html=True)
        with page_next:
            if st.button("다음 페이지"):
                # PDF 파일의 총 페이지 수를 동적으로 가져오기 어려우므로, 최대 페이지 수를 임의로 설정
                if st.session_state.pdf_page_number < 17:
                    st.session_state.pdf_page_number += 1
                    st.rerun()
        
        pdf_viewer(
            input=pdf_file_data, 
            pages_to_render=[st.session_state.pdf_page_number]
        )
    else:
        st.warning(f"'{pdf_path}' 경로에서 PDF 파일을 찾을 수 없습니다. 인덱싱을 진행하면 파일이 생성됩니다.")

with col2:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 인덱스 존재 여부 확인
    index_exists = True
    try:
        opensearch_client.indices.get(index=opensearch_index_name)
    except NotFoundError:
        index_exists = False
    
    # 인덱스가 존재하지 않을 때만 인덱싱 버튼 표시
    if not index_exists:
        st.warning(f"OpenSearch 인덱스 '{opensearch_index_name}'이(가) 존재하지 않습니다. PDF 인덱싱을 먼저 진행해야 합니다.")
        if st.button("PDF 인덱싱 시작"):
            with st.spinner("PDF를 인덱싱하고 있습니다. 잠시만 기다려주세요..."):
                s3_file_key = "accu.pdf"
                process_pdf_and_index_opensearch_with_images(s3_file_key)
                st.success("PDF 인덱싱이 완료되었습니다. 이제 챗봇에 질문해 보세요.")
                st.rerun()

    # 인덱스가 존재할 때만 챗봇 UI 표시
    if index_exists:
        st.success(f"OpenSearch 인덱스 '{opensearch_index_name}'이(가) 이미 존재합니다. 바로 질문해 보세요.")
        
        st.markdown("---")
        st.markdown("## 샘플 질문 (클릭하고 맨 아래 질문 보내기 버튼 클릭))")

        sample_questions = [
            "프로젝트 관리 페이지에서 어떤 작업을 할 수 있습니까?",
            "사용자 이관 기능은 어떤 계정만 가능합니까?",
            "프로젝트 관리에서 특정 프로젝트 수정을 하는 방법은? 순서대로 알려주세요."
        ]

        # 샘플 질문 버튼을 클릭하면 st.session_state에 값을 저장
        cols = st.columns(2)
        for i, q in enumerate(sample_questions):
            with cols[i % 2]:
                if st.button(q, key=f"btn_{i}"):
                    st.session_state.text_input_value = q
                    st.rerun()
        st.write("---")
        
        # 채팅 기록을 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "images" in message:
                    for image_path in message["images"]:
                        st.image(image_path)

    # 텍스트 입력창과 전송 버튼
    if index_exists:
        with st.form(key='chat_form', clear_on_submit=True):
            submit_button_label = "전송"
            if st.session_state.text_input_value:
                submit_button_label = "🌟 질문 보내기"
            
            user_input = st.text_input(
                "질문을 입력하세요...", 
                value=st.session_state.text_input_value, 
                key='user_input_key',
                label_visibility='hidden'
            )
            submit_button = st.form_submit_button(submit_button_label)
        
        if submit_button and user_input:
            st.session_state.text_input_value = "" # 입력 후 값 초기화
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            
            with st.chat_message("assistant"):
                with st.spinner("답변을 생성 중입니다..."):
                    answer, image_paths = get_rag_answer_from_bedrock_with_images(user_input)
                    st.markdown(answer)
                    if image_paths:
                        for image_path in image_paths:
                            st.image(os.path.join(FIGURES_DIR, image_path), caption=os.path.basename(image_path))
            
            st.session_state.messages.append({"role": "assistant", "content": answer, "images": [os.path.join(FIGURES_DIR, p) for p in image_paths]})
            st.rerun()