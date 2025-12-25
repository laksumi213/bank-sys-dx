import streamlit as st
import pandas as pd
import os
import glob
import hashlib
import base64
from datetime import datetime
from dotenv import load_dotenv

# --- 安定性を重視した最新のインポート ---
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 💡 ポイント：エラーの元となる langchain.chains 関連のインポートを完全に排除しました

# 1. 環境設定
load_dotenv()
st.set_page_config(page_title="銀行手続DXポータル", layout="wide")

# パス設定
DB_FILE = "bank_document_index.xlsx"
DATA_DIR = "./clients_data"
CHROMA_DIR = "./chroma_db"

# フォルダが存在しない場合は作成
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# --- ユーティリティ関数 ---

def get_file_hash(path):
    """ファイルの中身からハッシュ値を計算（変更検知用）"""
    hasher = hashlib.md5()
    with open(path, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def get_pdf_display_link(file_path):
    """PDFをStreamlit上に埋め込むためのHTML生成"""
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    return f'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf">'

# --- メインロジック：データ同期 ---

def sync_data():
    st.info("スキャンを開始します。スキャン画像を含むため、AIの視覚解析モードを使用します...")
    
    # 1. 全PDFファイルを再帰的に検索
    pdf_files = glob.glob(os.path.join(DATA_DIR, "**/*.pdf"), recursive=True)
    
    # 2. 既存台帳の読み込み
    if os.path.exists(DB_FILE):
        df_db = pd.read_excel(DB_FILE)
    else:
        df_db = pd.DataFrame(columns=["ファイル名", "フルパス", "ハッシュ", "銀行", "書類種別", "最終更新日"])

    # 3. 解析モデルの準備（Gemini 1.5 Flashは画像/PDFを直接読めます）
    # APIキーの適用
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("gemini-2.5-flash")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 4. ファイル精査
    for file_path in pdf_files:
        file_hash = get_file_hash(file_path)
        
        is_new = file_path not in df_db["フルパス"].values
        is_modified = not is_new and df_db.loc[df_db["フルパス"] == file_path, "ハッシュ"].values[0] != file_hash
        
        if is_new or is_modified:
            st.write(f"👁️ AI視覚解析中: {os.path.basename(file_path)}")
            
            try:
                # 【重要】PDFをGeminiに直接アップロード（これでスキャン画像も読める）
                # MIMEタイプを指定してファイルをアップロード
                uploaded_file = genai.upload_file(file_path, mime_type="application/pdf")
                
                # プロンプト：JSON形式で情報を抽出させる
                prompt = """
                この銀行手続書類を解析し、以下の情報を抽出してください。
                出力は必ずカンマ区切りのテキストのみ（CSV形式）で行ってください。余計な文章は不要です。
                
                フォーマット:
                銀行名,書類の種類,書類の要約テキスト(全文のOCR結果)
                
                例:
                三菱UFJ銀行,残高証明書発行依頼書,被相続人〇〇の残高証明を依頼する書類。実印が必要。
                """
                
                # Geminiに画像を見せて回答させる
                response = model.generate_content([prompt, uploaded_file])
                
                # 解析結果の分割
                parts = response.text.split(",", 2)
                if len(parts) >= 3:
                    bank = parts[0].strip()
                    doc_type = parts[1].strip()
                    summary_text = parts[2].strip()
                else:
                    bank, doc_type, summary_text = "解析エラー", "解析エラー", response.text

                # 台帳更新
                new_row = {
                    "ファイル名": os.path.basename(file_path),
                    "フルパス": file_path,
                    "ハッシュ": file_hash,
                    "銀行": bank,
                    "書類種別": doc_type,
                    "最終更新日": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                df_db = df_db[df_db["フルパス"] != file_path]
                df_db = pd.concat([df_db, pd.DataFrame([new_row])], ignore_index=True)
                
                # 【重要】スキャンデータだとPyPDFLoaderではテキストが取れないため、
                # Geminiが目で見て書き起こした「summary_text」をベクターDBに入れる
                from langchain.schema import Document
                doc = Document(page_content=summary_text, metadata={"source": file_path, "bank": bank})
                
                # テキスト分割して保存
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                splits = text_splitter.split_documents([doc])
                
                # 最新のインポートで警告対応済み
                from langchain_chroma import Chroma
                Chroma.from_documents(
                    documents=splits,
                    embedding=embeddings,
                    persist_directory=CHROMA_DIR
                )
                
            except Exception as e:
                st.error(f"解析失敗 ({os.path.basename(file_path)}): {e}")
            
    # クリーニングと保存
    df_db = df_db[df_db["フルパス"].apply(os.path.exists)]
    df_db.to_excel(DB_FILE, index=False)
    return df_db

# --- 画面レイアウト ---

st.title("🏦 銀行手続ナレッジ共有システム")

# サイドバー：管理機能
with st.sidebar:
    st.header("⚙️ システム管理")
    if st.button("🔄 フォルダを同期してAI解析"):
        df_result = sync_data()
        st.success("同期が完了しました。")
        st.rerun()
    
    st.divider()
    st.caption("※ clients_data フォルダ内の全PDFをスキャンします。")

# 台帳データの読み込み
if os.path.exists(DB_FILE):
    df = pd.read_excel(DB_FILE)
else:
    df = pd.DataFrame()

# メインコンテンツ
if not df.empty:
    tab1, tab2 = st.tabs(["🔍 書類検索・プレビュー", "💬 AI書き方相談"])

    # --- TAB1: 検索とプレビュー ---
    with tab1:
        search_q = st.text_input("銀行名や書類名、人名で検索", placeholder="例: 三菱UFJ 残高証明")
        
        # 検索フィルタリング
        mask = df.apply(lambda row: search_q.lower() in str(row).lower(), axis=1)
        filtered_df = df[mask]
        
        if not filtered_df.empty:
            col_list, col_view = st.columns([1, 1.2])
            
            with col_list:
                st.write(f"検索結果: {len(filtered_df)} 件")
                selected_file_name = st.selectbox("確認する書類を選択", filtered_df["ファイル名"].tolist())
                selected_row = filtered_df[filtered_df["ファイル名"] == selected_file_name].iloc[0]
                
                # 簡易台帳編集機能
                st.info(f"📍 パス: {selected_row['フルパス']}")
                with st.expander("台帳情報を修正する"):
                    new_bank = st.text_input("銀行名", selected_row["銀行"])
                    new_type = st.text_input("書類種別", selected_row["書類種別"])
                    if st.button("修正を保存"):
                        df.loc[df["フルパス"] == selected_row["フルパス"], ["銀行", "書類種別"]] = [new_bank, new_type]
                        df.to_excel(DB_FILE, index=False)
                        st.toast("修正を台帳に反映しました")
            
            with col_view:
                st.markdown(get_pdf_display_link(selected_row["フルパス"]), unsafe_allow_html=True)
        else:
            st.warning("条件に一致する書類がありません。")

    # --- TAB2: AIコンサルタント ---
    with tab2:
        st.subheader("🤖 AI書き方コンサルタント")
        
        # ベクターDBのロード
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        if os.path.exists(CHROMA_DIR):
            vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
            chat_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

            # チャットインターフェース
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for m in st.session_state.messages:
                with st.chat_message(m["role"]): st.markdown(m["content"])

            if user_input := st.chat_input("例: この銀行の残高証明で、代理人の住所はどこに書けばいい？"):
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"): st.markdown(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("過去の書類を照合中..."):
                        # 💡 修正ポイント：RetrievalQAなどのChainを使わず、手動でRAGを完結
                        # 1. 関連ドキュメントを検索
                        docs = vectorstore.similarity_search(user_input, k=5)
                        context = "\n\n".join([d.page_content for d in docs])
                        
                        # 2. Gemini用のプロンプトを作成
                        full_prompt = f"""あなたは行政書士の実務補助者です。以下の過去事例（コンテキスト）を参考に、質問に回答してください。
                        コンテキストにない情報は「不明」と答え、知ったかぶりをしないでください。
                        
                        コンテキスト:
                        {context}
                        
                        質問: {user_input}"""
                        
                        # 3. 直接LLMを呼び出す
                        response = chat_llm.invoke(full_prompt)
                        answer = response.content
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.info("サイドバーの『同期』ボタンを押して、AIの知識ベースを構築してください。")
else:
    st.info("clients_data フォルダにPDFファイルを入れて、サイドバーの『同期』ボタンを押してください。")