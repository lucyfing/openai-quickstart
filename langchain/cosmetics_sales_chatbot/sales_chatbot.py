import gradio as gr

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain import LLMChain

import os
os.environ['OPENAI_API_KEY'] = 'XXX'
os.environ['OPENAI_BASE_URL'] = 'XXX'


def initialize_sales_bot(vector_store_dir: str="real_cosmetics_sales"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

    global SALES_BOT    
    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                           retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                     search_kwargs={"score_threshold": 0.8}))
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT


def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
   # TODO: 从命令行参数中获取
    enable_chat = True

    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if len(ans["source_documents"]) > 0:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        print(len(ans["source_documents"]))
        return ans["result"]
    # 否则输出套路话术
    else:
        template = """你是一位化妆品销售。根据传入的问题，对问题做出回答。
                    问题：{question}
                    答案：以下是对上述问题的回答："""
        fallback_prompt = PromptTemplate(
            input_variables=["message"],
            template=template
        )
        fallback_chain = LLMChain(prompt=fallback_prompt, llm=llm, output_key="fallback", verbose=True)
        return fallback_chain.invoke(message).get('fallback', '').strip()
    

def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="化妆品销售",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=500),
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.5)
    # 初始化化妆品销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
