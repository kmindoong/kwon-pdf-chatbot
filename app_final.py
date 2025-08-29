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
from unstructured.documents.elements import Title, NarrativeText, ListItem, Image
from io import BytesIO
from opensearchpy.exceptions import NotFoundError, ConnectionError
from streamlit_pdf_viewer import pdf_viewer
import fitz  # PyMuPDF 라이브러리
from PIL import Image as PilImage # 이미지 처리를 위해 필요
import pytesseract

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
opensearch_index_name = os.getenv("OPENSEARCH_KWON_NAME2")
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
PDF_DIR = "./multi-pdf-files"
FIGURES_DIR = "./images"
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ========================
# 2. 인덱싱 함수 (텍스트 + 이미지) - 다중 파일 처리
# ========================

# 현재 코드에서는 unstructured 라이브러리를 사용하고 있습니다. unstructured는 다양한 문서 형식에 범용적으로 사용되지만, PDF 파싱에 특화된 라이브러리를 사용하면 더 빠르고 정확한 결과를 얻을 수 있습니다.
# PyMuPDF (fitz): 속도와 정확성이 매우 뛰어난 라이브러리입니다. PDF의 텍스트와 레이아웃 정보를 빠르게 추출합니다.
# pdfplumber: 표(table) 추출 기능이 강력하며, 특정 영역의 텍스트나 이미지를 선택적으로 파싱하는 데 유용합니다.
# PyMuPDF는 PDF의 내부에 포함된 텍스트를 추출하는 데 매우 효과적이지만, **이미지에 포함된 텍스트(OCR)**를 직접 추출하는 기능은 기본적으로 제공하지 않습니다.
# 따라서 PyMuPDF를 사용하면서 이미지 내 텍스트를 함께 추출하려면, OCR 엔진을 별도로 연동해야 합니다.
# unstructured 라이브러리는 tesseract-ocr과 같은 OCR 엔진을 백엔드로 사용하여 이미지에서 텍스트를 추출하는 기능을 제공합니다. 현재 코드가 이미지의 텍스트를 제대로 인식하지 못하는 이유는 다음과 같을 수 있습니다.
# Tesseract OCR 엔진 미설치: unstructured 라이브러리가 의존하는 tesseract-ocr이 EC2 인스턴스에 설치되어 있지 않을 수 있습니다.
# 언어 팩 부족: Tesseract가 한글을 인식하려면 한글 언어 팩이 설치되어 있어야 합니다.
# 낮은 해상도: PDF 문서의 이미지가 너무 저해상도인 경우, OCR 성능이 저하될 수 있습니다.


def process_pdfs_and_index_opensearch():
    try:
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
        if not pdf_files:
            st.error(f"'{PDF_DIR}' 폴더에 PDF 파일이 없습니다.")
            return

        logging.info("OpenSearch 인덱스 삭제 및 생성 중...")
        if opensearch_client.indices.exists(index=opensearch_index_name):
            opensearch_client.indices.delete(index=opensearch_index_name)
            logging.info(f"기존 인덱스 '{opensearch_index_name}'을 성공적으로 삭제했습니다.")
        
        mapping = {
            "settings": {
                "index.knn": True,
            },
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 1536,
                        "method": {
                            "name": "hnsw",
                            "engine": "nmslib"
                        }
                    },
                    "image_paths": {"type": "keyword"},
                    "page_number": {"type": "long"},
                    "source": {"type": "keyword"}
                }
            }
        }
        opensearch_client.indices.create(index=opensearch_index_name, body=mapping)
        logging.info(f"OpenSearch 인덱스 '{opensearch_index_name}' 생성 완료.")

        docs = []
        for file_name in pdf_files:
            logging.info(f"'{file_name}' 파일 처리 중...")
            local_pdf_path = os.path.join(PDF_DIR, file_name)

            elements = partition_pdf(
                filename=local_pdf_path,
                strategy="hi_res",
                infer_table_structure=False,
                languages=["kor", "eng"],
                ocr_languages="kor+eng",
                extract_images_in_pdf=True,
                image_output_dir_path=FIGURES_DIR
            )
            
            chunks = []
            current_chunk_text = ""
            current_chunk_images = []
            current_chunk_page = 1 # <-- 파일당 페이지 번호 초기화
            
            for element in elements:
                # 페이지 번호 업데이트
                if hasattr(element.metadata, 'page_number') and element.metadata.page_number:
                    current_chunk_page = element.metadata.page_number
                
                # 이미지 처리
                if isinstance(element, Image):
                    if hasattr(element.metadata, 'image_path'):
                        current_chunk_images.append(os.path.basename(element.metadata.image_path))
                    # OCR 텍스트는 unstructured가 자동으로 element.text에 포함합니다.
                    if element.text and len(element.text) > 20:
                         current_chunk_text += "\n\n이미지 설명: " + element.text
                         
                # 제목을 기준으로 청크 분리
                elif isinstance(element, Title):
                    if current_chunk_text.strip():
                        chunks.append({
                            "text": current_chunk_text.strip(),
                            "image_paths": current_chunk_images,
                            "page_number": current_chunk_page
                        })
                    current_chunk_text = element.text
                    current_chunk_images = []
                # 일반 텍스트 추가
                elif isinstance(element, (NarrativeText, ListItem)):
                    current_chunk_text += "\n" + element.text
            
            # 마지막 청크 추가
            if current_chunk_text.strip():
                chunks.append({
                    "text": current_chunk_text.strip(),
                    "image_paths": current_chunk_images,
                    "page_number": current_chunk_page
                })
            
            for chunk in chunks:
                if not chunk['text']:
                    continue
                
                bedrock_model_id = 'amazon.titan-embed-text-v1'
                response = bedrock_client.invoke_model(
                    modelId=bedrock_model_id,
                    body=json.dumps({"inputText": chunk['text']})
                )
                embedding = json.loads(response['body'].read())['embedding']

                doc = {
                    'text': chunk['text'],
                    'source': file_name,
                    'page_number': chunk['page_number'],
                    'embedding': embedding,
                    'image_paths': chunk['image_paths']
                }
                docs.append({
                    '_index': opensearch_index_name,
                    '_source': doc
                })

        if docs:
            try:
                successes, failures = bulk(opensearch_client, docs)
                if failures:
                    logging.error(f"인덱싱 실패: {failures}")
                else:
                    logging.info(f"OpenSearch에 성공적으로 인덱싱되었습니다. 총 문서 수: {len(docs)}")
            except ConnectionError as e:
                logging.error(f"벌크 인덱싱 중 연결 오류: {e}")
            except Exception as e:
                logging.error(f"벌크 인덱싱 중 오류 발생: {e}")
        else:
            logging.warning("인덱싱할 문서가 없습니다.")
            
    except Exception as e:
        logging.error(f"오류 발생: {e}")

def get_rag_answer_from_bedrock_with_images(query):
    try:
        # LLM 모델을 사용하여 쿼리 확장 (Haiku 모델 사용)
        llm_model_id_expansion = 'anthropic.claude-3-haiku-20240307-v1:0'
        expansion_prompt = f"""
        주어진 질문을 더 구체적이고 풍부한 검색 쿼리로 확장해줘.
        단, 원래 질문의 핵심 키워드를 반드시 포함해야 해.
        
        예시:
        - 원본 질문: "로그인 계정은?"
        - 확장 쿼리: "Azure 자격증 신청 시 로그인 계정"

        원본 질문: {query}
        확장 쿼리: 
        """
        body_expansion = json.dumps({
            "anthropic_version": "bedrock-2023-05-31", 
            "max_tokens": 100, 
            "messages": [{"role": "user", "content": [{"type": "text", "text": expansion_prompt}]}], 
            "temperature": 0.5
        })
        response_body_expansion = bedrock_client.invoke_model(
            modelId=llm_model_id_expansion, 
            body=body_expansion, 
            contentType="application/json", 
            accept="application/json"
        )
        expanded_query = json.loads(response_body_expansion.get('body').read())['content'][0]['text']
        
        # 확장된 쿼리를 사용하여 임베딩 생성
        bedrock_model_id_embedding = 'amazon.titan-embed-text-v1'
        response_embedding = bedrock_client.invoke_model(
            modelId=bedrock_model_id_embedding,
            body=json.dumps({"inputText": expanded_query})
        )
        query_embedding = json.loads(response_embedding['body'].read())['embedding']
        
        # 하이브리드 검색 튜닝: 확장된 쿼리에 높은 가중치를 부여
        search_query = {
            "size": 3,
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": expanded_query,
                                "fields": ["text"] # 확장된 쿼리에 3배 가중치 부여
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
        
        # LLM 모델 호출
        llm_model_id = 'anthropic.claude-3-haiku-20240307-v1:0'
        
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
    st.markdown("## PDF 미리보기 뷰어 (첫번째 파일만 보이지롱)", unsafe_allow_html=True)
    
    # multi-pdf-files 폴더에서 PDF 파일 목록 가져오기
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    
    if pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf_files[0]) # 첫 번째 파일만 미리보기
        with open(pdf_path, "rb") as f:
            pdf_file_data = f.read()

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
                if st.session_state.pdf_page_number < 10: # 최대 10페이지 미리보기
                    st.session_state.pdf_page_number += 1
                    st.rerun()
        
        pdf_viewer(
            input=pdf_file_data, 
            pages_to_render=[st.session_state.pdf_page_number]
        )
    else:
        st.warning(f"'{PDF_DIR}' 폴더에 PDF 파일을 찾을 수 없습니다. 인덱싱을 진행하면 파일이 생성됩니다.")

with col2:
    index_exists = True
    try:
        opensearch_client.indices.get(index=opensearch_index_name)
    except NotFoundError:
        index_exists = False
    
    # 인덱싱 시작 버튼 로직 수정
    if not index_exists:
        st.warning(f"OpenSearch 인덱스 '{opensearch_index_name}'이(가) 존재하지 않습니다. PDF 인덱싱을 먼저 진행해야 합니다.")

        # 인덱싱 작업 중인지 세션 상태에 저장
        if 'is_indexing' not in st.session_state:
            st.session_state.is_indexing = False
        
        # 버튼을 클릭하면 is_indexing 상태를 True로 변경
        if st.button("PDF 인덱싱 시작", disabled=st.session_state.is_indexing):
            st.session_state.is_indexing = True
            st.rerun()

        # is_indexing 상태가 True일 때만 프로세스 실행 및 스피너 표시
        if st.session_state.is_indexing:
            with st.spinner("PDF를 인덱싱하고 있습니다. 잠시만 기다려주세요..."):
                process_pdfs_and_index_opensearch()
                st.session_state.is_indexing = False # 인덱싱 완료 후 상태 초기화
                st.success("🎉 PDF 인덱싱이 완료되었습니다. 이제 챗봇에 질문해 보세요.")
                st.rerun()

    else:
        st.success(f"OpenSearch 인덱스 '{opensearch_index_name}'이(가) 이미 존재합니다. 바로 질문해 보세요.")
        
        st.markdown("---")
        st.markdown("## 샘플 질문 (클릭하고 맨 아래 질문 보내기 버튼 클릭)")

        sample_questions = [
            "역량 self profiling의 Skill set 단계가 무엇이 있으며 각 단계별 의미를 알려줘",
            "역량 self profiling 문의가 생기면 어디다가 연락하면 돼?",
            "역량 self profiling 도입 배경은?",
            "직무 등급 결과는 언제 오픈 되니?",
            "팀원의 Skill set 조회와 직무는 어디서 확인할 수 있나?",
            "롯데온의 판매가격 및 재고수량 설정 방법을 단계별로 알려줘.",
            "롯데온의 배송/반품정보에서 배송정보 설정 방법 알려줘.",
            "롯데온의 FAQ를 알려줘.",
            "Azure 자격증 시험 신청 방법 알려줘.",
            "Azure 자격증 획득 후 MPN 연동 방법 알려줘.",
            "Azure 자격증 신청할 때 로그인 계정은?"
        ]
        
        cols = st.columns(2)
        for i, q in enumerate(sample_questions):
            with cols[i % 2]:
                if st.button(q, key=f"btn_{i}"):
                    st.session_state.text_input_value = q
                    st.rerun()
        st.write("---")
        
        # 변경된 부분: 이전 대화 기록은 expander로, 마지막 대화는 그대로 표시
        if len(st.session_state.messages) > 2:
            for i in range(0, len(st.session_state.messages) - 2, 2):
                user_message = st.session_state.messages[i]
                assistant_message = st.session_state.messages[i+1]
                
                with st.expander(f"Q: {user_message['content'][:50]}..."):
                    st.markdown(f"**질문**: {user_message['content']}")
                    st.markdown(f"**답변**: {assistant_message['content']}")
                    if "images" in assistant_message and assistant_message["images"]:
                        for image_path in assistant_message["images"]:
                            st.image(image_path)
        
        # 마지막 대화만 따로 표시 (자동으로 접히지 않음)
        if st.session_state.messages:
            last_user_message = st.session_state.messages[-2]
            last_assistant_message = st.session_state.messages[-1]
            
            with st.chat_message(last_user_message["role"]):
                st.markdown(last_user_message["content"])
            with st.chat_message(last_assistant_message["role"]):
                st.markdown(last_assistant_message["content"])
                if "images" in last_assistant_message and last_assistant_message["images"]:
                    for image_path in last_assistant_message["images"]:
                        st.image(image_path)

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
            st.session_state.text_input_value = ""
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