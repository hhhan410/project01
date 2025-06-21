import json
from googletrans import Translator
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datetime import datetime


class TranslationAssistant:
    def __init__(self):
        # 初始化翻译器和问答模型
        self.translator = Translator()
        self.initialize_qa_model()
        self.history = []

    def initialize_qa_model(self):
        """初始化一个小型问答模型"""
        print("正在加载问答模型...")
        # 使用一个轻量级的问答模型
        self.qa_pipeline = pipeline("question-answering",
                                    model="distilbert-base-cased-distilled-squad")
        # 加载聊天模型用于对话
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        self.chat_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        self.chat_history_ids = None
        print("问答模型加载完成")

    def translate_text(self, text, target_lang="en", source_lang=None):
        """将文本从源语言翻译到目标语言"""
        try:
            if not source_lang:
                # 自动检测源语言
                translation = self.translator.translate(text, dest=target_lang)
            else:
                translation = self.translator.translate(text, src=source_lang, dest=target_lang)

            result = {
                "source_text": translation.origin,
                "translated_text": translation.text,
                "source_language": translation.src,
                "target_language": translation.dest
            }

            self.history.append({"type": "translation", "input": text,
                                 "result": result, "timestamp": datetime.now().isoformat()})

            return result
        except Exception as e:
            return {"error": f"翻译失败: {str(e)}"}

    def answer_question(self, question, context=None):
        """回答关于翻译的问题或进行对话"""
        if context:
            # 基于给定上下文的问答
            result = self.qa_pipeline(question=question, context=context)
            answer = f"答案: {result['answer']} (置信度: {result['score']:.4f})"
        else:
            # 通用对话
            new_user_input_ids = self.tokenizer.encode(question + self.tokenizer.eos_token, return_tensors='pt')

            # 如果有对话历史，将其与新问题连接
            if self.chat_history_ids is not None:
                bot_input_ids = torch.cat([self.chat_history_ids, new_user_input_ids], dim=-1)
            else:
                bot_input_ids = new_user_input_ids

            # 生成回复
            self.chat_history_ids = self.chat_model.generate(
                bot_input_ids, max_length=1000,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # 解码回复
            answer = self.tokenizer.decode(
                self.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
                skip_special_tokens=True
            )

        self.history.append({"type": "qa", "question": question, "answer": answer,
                             "timestamp": datetime.now().isoformat()})
        return answer

    def get_supported_languages(self):
        """获取支持的语言列表"""
        return list(self.translator.LANGUAGES.values())

    def save_history(self, filename="translation_history.json"):
        """保存交互历史到文件"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)


# 使用示例
if __name__ == "__main__":
    assistant = TranslationAssistant()

    # 翻译示例
    translation = assistant.translate_text("你好，世界！", target_lang="en")
    print(f"翻译结果: {translation['translated_text']}")

    # 问答示例
    context = """
    翻译是将一种自然语言的文本转换为另一种自然语言的过程。
    常见的翻译工具有Google翻译、百度翻译和DeepL等。
    翻译时需要注意语言的语法结构、文化背景和习语等因素。
    """

    question = "什么是翻译?"
    answer = assistant.answer_question(question, context)
    print(f"\n问题: {question}\n回答: {answer}")

    # 对话示例
    print("\n=== 开始对话 ===")
    while True:
        user_input = input("你: ")
        if user_input.lower() in ["退出", "bye", "quit"]:
            break
        response = assistant.answer_question(user_input)
        print(f"助手: {response}")

    # 保存历史记录
    assistant.save_history()
    print("\n历史记录已保存")