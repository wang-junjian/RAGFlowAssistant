import streamlit as st
import os
import time
import uuid
from ragflow_sdk import RAGFlow
from dotenv import load_dotenv, set_key

# 确保 st.set_page_config() 是第一个 Streamlit 命令
st.set_page_config(
    page_title="RAGFlow 聊天助手",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 使用 Streamlit UI 库开发一个连接 RAGFlow 的客户端应用，左边列出可选的知识库，右边是聊天对话框。

def init_session_state():
    """初始化所有需要的会话状态变量"""
    # API 配置相关
    if "ragflow_api_key" not in st.session_state:
        st.session_state.ragflow_api_key = ""
    if "ragflow_base_url" not in st.session_state:
        st.session_state.ragflow_base_url = "http://localhost:9380"
    
    # 聊天相关
    if "active_chat" not in st.session_state:
        st.session_state.active_chat = None
    if "active_session" not in st.session_state:
        st.session_state.active_session = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 知识库相关
    if "selected_datasets" not in st.session_state:
        st.session_state.selected_datasets = []
    if "available_datasets" not in st.session_state:
        st.session_state.available_datasets = []

def init_ragflow():
    """初始化 RAGFlow 客户端"""
    # 加载环境变量
    load_dotenv(override=True)  # 添加 override=True 确保每次都重新加载
    
    # 优先使用环境变量，如果不存在则使用会话状态中的值
    api_key = os.environ.get("RAGFLOW_API_KEY") or st.session_state.get("ragflow_api_key", "")
    base_url = os.environ.get("RAGFLOW_BASE_URL") or st.session_state.get("ragflow_base_url", "http://localhost:9380")
    
    # 更新会话状态
    st.session_state.ragflow_api_key = api_key
    st.session_state.ragflow_base_url = base_url
    
    if api_key:
        return RAGFlow(api_key=api_key, base_url=base_url)
    return None

def create_new_chat():
    """创建新的聊天对话，重置会话状态"""
    # 清除消息历史
    st.session_state.messages = []
    
    # 重置聊天和会话
    st.session_state.active_chat = None
    st.session_state.active_session = None
    
    # 保留已选择的知识库
    # 添加通知
    st.toast("已创建新的聊天对话", icon="🔄")

def sidebar_content(rag_object):
    # 添加新建聊天按钮
    if st.sidebar.button("🔄 新建聊天对话", use_container_width=True):
        create_new_chat()
    
    st.sidebar.title("RAGFlow 知识库")
    
    st.sidebar.divider()
        
    if rag_object:
        try:
            datasets = rag_object.list_datasets()
            st.session_state.available_datasets = datasets
            
            selected_datasets = []
            for dataset in datasets:
                if st.sidebar.checkbox(f"{dataset.name}", key=f"dataset_{dataset.id}"):
                    selected_datasets.append(dataset.id)
            
            st.session_state.selected_datasets = selected_datasets
            
            if selected_datasets:
                st.sidebar.success(f"已选择 {len(selected_datasets)} 个知识库")
            else:
                st.sidebar.info("请至少选择一个知识库")
                
        except Exception as e:
            st.sidebar.error(f"获取知识库失败: {str(e)}")
    else:
        st.sidebar.warning("请先提供有效的 RAGFlow API Key")

    st.sidebar.divider()

    # 使用 expander 创建可折叠的配置区域
    with st.sidebar.expander("⚙️ 配置", expanded=False):
        # 添加 API Key 配置
        api_key = st.text_input(
            "RAGFlow API Key",
            value=st.session_state.ragflow_api_key,
            type="password",
            help="请输入您的 RAGFlow API Key"
        )
        if api_key != st.session_state.ragflow_api_key:
            st.session_state.ragflow_api_key = api_key
            # 更新环境变量
            set_key(".env", "RAGFLOW_API_KEY", api_key)
            st.success("API Key 已更新")
            st.rerun()

        # 添加 Base URL 配置
        base_url = st.text_input(
            "RAGFlow Base URL",
            value=st.session_state.ragflow_base_url,
            help="RAGFlow 服务器地址"
        )
        if base_url != st.session_state.ragflow_base_url:
            st.session_state.ragflow_base_url = base_url
            # 更新环境变量
            set_key(".env", "RAGFLOW_BASE_URL", base_url)
            st.success("Base URL 已更新")
            st.rerun()

def generate_unique_name(prefix="Chat"):
    """生成唯一的聊天或会话名称"""
    unique_id = str(uuid.uuid4())[:8]
    timestamp = int(time.time())
    return f"{prefix}_{timestamp}_{unique_id}"

def format_reference(ref):
    """将不同格式的引用转换为统一的字典格式"""
    if isinstance(ref, dict):
        return ref
    else:
        # 如果是对象，转换为字典
        return {
            "document_name": getattr(ref, "document_name", "未知文档"),
            "similarity": getattr(ref, "similarity", 0.0),
            "content": getattr(ref, "content", ""),
            "document_id": getattr(ref, "document_id", "")
        }

def process_stream_response(session, prompt, message_placeholder):
    """处理流式响应并返回完整响应和引用"""
    try:
        full_response = ""
        response = None
        
        # 流式展示回答
        for response in session.ask(prompt, stream=True):
            new_content = response.content[len(full_response):]
            full_response = response.content
            message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
        
        # 处理引用
        references = []
        if hasattr(response, "reference") and response.reference:
            references = [format_reference(ref) for ref in response.reference]
        
        return full_response, references
    except Exception as e:
        message_placeholder.error(f"获取回答时出错: {str(e)}")
        return f"抱歉，处理您的问题时出现错误: {str(e)}", []

def main_content(rag_object):
    """渲染主内容区域，展示聊天界面"""
    # 标题
    st.title("RAGFlow 聊天助手")
    
    # 显示当前会话状态
    if st.session_state.active_chat and st.session_state.active_session:
        st.caption(f"当前对话：{getattr(st.session_state.active_chat, 'name', '未命名对话')}")
    
    # 显示历史消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # 如果消息有引用，显示引用
            if "reference" in message and message["reference"]:
                with st.expander("查看引用来源"):
                    for ref in message["reference"]:
                        if isinstance(ref, dict):
                            st.markdown(f"**来源文档**: {ref.get('document_name', '未知文档')}")
                            st.markdown(f"**内容**: {ref.get('content', '无内容')}")
                        else:
                            st.markdown(f"**来源文档**: {getattr(ref, 'document_name', '未知文档')}")
                            st.markdown(f"**内容**: {getattr(ref, 'content', '无内容')}")
                        st.markdown("---")
    
    # 聊天输入
    if prompt := st.chat_input("输入您的问题..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 创建或使用现有的聊天会话
        if rag_object and st.session_state.selected_datasets:
            with st.chat_message("assistant"):
                try:
                    # 如果还没有活跃的聊天或会话，创建一个
                    if not st.session_state.active_chat:
                        # 使用唯一名称创建聊天
                        chat_name = generate_unique_name("Chat")
                        chat = rag_object.create_chat(
                            name=chat_name, 
                            dataset_ids=st.session_state.selected_datasets
                        )
                        st.session_state.active_chat = chat
                        
                        # 使用唯一名称创建会话
                        session_name = generate_unique_name("Session")
                        session = chat.create_session(session_name)
                        st.session_state.active_session = session
                    else:
                        session = st.session_state.active_session
                    
                    # 使用流式响应
                    message_placeholder = st.empty()
                    
                    # 处理流式响应
                    full_response, references = process_stream_response(
                        session, prompt, message_placeholder
                    )
                    
                    # 将助手响应添加到会话状态
                    assistant_message = {"role": "assistant", "content": full_response}
                    if references:
                        assistant_message["reference"] = references
                    
                    st.session_state.messages.append(assistant_message)
                    
                    # 添加重新运行，强制刷新UI以显示最新消息
                    st.rerun()
                    
                except Exception as e:
                    error_msg = str(e)
                    if "Duplicated chat name" in error_msg:
                        st.error("创建聊天失败：聊天名称重复，请尝试刷新页面")
                    else:
                        st.error(f"聊天出错: {error_msg}")
                        st.error(f"错误详情: {type(e).__name__}")
        else:
            with st.chat_message("assistant"):
                st.warning("请先选择至少一个知识库，并确保 RAGFlow 连接正常")
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "请先选择至少一个知识库，并确保 RAGFlow 连接正常"
                })


def main():
    """主函数"""
    # 初始化会话状态
    init_session_state()
    
    # 初始化 RAGFlow 客户端
    rag_object = init_ragflow()
    
    # 渲染侧边栏
    sidebar_content(rag_object)
    
    # 渲染主内容
    main_content(rag_object)

if __name__ == "__main__":
    main()
