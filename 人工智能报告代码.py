import os
import sys
import logging
import argparse
import hashlib
import numpy as np
import requests
from typing import Any, Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("translation_qa.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("TranslationQA")

# 语言代码映射表（适配百度API）
LANGUAGE_MAP = {
    "中文": "zh",
    "英文": "en",
    "日文": "jp",
    "韩文": "kor",
    "法文": "fra",
    "德文": "de",
    "西班牙文": "spa",
    "俄文": "ru",
    "阿拉伯文": "ara",
    "葡萄牙文": "pt",
    "auto": "auto"  # 自动检测语言
}


@dataclass
class TranslationResult:
    """翻译结果数据类"""
    source_text: str
    source_language: str
    target_language: str
    translated_text: str
    confidence: float
    translation_time: float
    model_used: str
    additional_info: Dict = None


@dataclass
class QAResult:
    """问答结果数据类"""
    question: str
    answer: str
    confidence: float
    source_context: str
    model_used: str = "rule"  # rule或qianfan


class TranslationModel:
    """翻译模型实现（使用百度翻译API）"""

    def __init__(self, appid: str = None, secret_key: str = None):
        self.appid = appid or os.environ.get("BAIDU_FANYI_APPID")
        self.secret_key = secret_key or os.environ.get("BAIDU_FANYI_SECRET_KEY")
        self.api_url = "http://api.fanyi.baidu.com/api/trans/vip/translate"
        self.language_codes = LANGUAGE_MAP
        self.proxies = self._get_proxies()  # 获取代理配置

        if not self.appid or not self.secret_key:
            logger.warning("未提供百度翻译API凭证，云翻译功能可能受限")
            logger.info("请通过环境变量或参数传入APP ID和Secret Key")

    def _get_proxies(self) -> Optional[Dict]:
        """获取代理配置（如需通过代理访问网络）"""
        http_proxy = os.environ.get("HTTP_PROXY", os.environ.get("http_proxy"))
        https_proxy = os.environ.get("HTTPS_PROXY", os.environ.get("https_proxy"))
        if http_proxy or https_proxy:
            return {
                "http": http_proxy,
                "https": https_proxy
            }
        return None

    def translate(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        """使用百度API翻译文本"""
        if not self.appid or not self.secret_key:
            return TranslationResult(
                source_text=text,
                source_language=source_lang,
                target_language=target_lang,
                translated_text="未配置百度翻译API凭证，无法进行云翻译",
                confidence=0.0,
                translation_time=0.0,
                model_used="baidu-translate"
            )

        start_time = datetime.now()

        # 处理语言代码映射
        name_to_code = {name: code for name, code in self.language_codes.items()}
        code_to_name = {code: name for name, code in self.language_codes.items()}

        from_lang = source_lang.lower()
        to_lang = target_lang.lower()

        if from_lang in name_to_code:
            from_lang = name_to_code[from_lang]
        if to_lang in name_to_code:
            to_lang = name_to_code[to_lang]

        if from_lang not in self.language_codes.values() or to_lang not in self.language_codes.values():
            error_msg = f"不支持的语言: 源语言={from_lang}, 目标语言={to_lang}"
            logger.error(error_msg)
            return TranslationResult(
                source_text=text,
                source_language=source_lang,
                target_language=target_lang,
                translated_text=error_msg,
                confidence=0.0,
                translation_time=0.0,
                model_used="baidu-translate"
            )

        # 生成签名
        salt = str(np.random.randint(32768, 65536))
        sign = self.appid + text + salt + self.secret_key
        sign = sign.encode("utf-8")
        sign = hashlib.md5(sign).hexdigest()

        # 发送请求（支持代理）
        try:
            response = requests.post(
                self.api_url,
                data={
                    "q": text,
                    "from": from_lang,
                    "to": to_lang,
                    "appid": self.appid,
                    "salt": salt,
                    "sign": sign
                },
                proxies=self.proxies,  # 应用代理
                timeout=10
            )
            response.raise_for_status()
            result = response.json()

            if "error_code" in result:
                error_msg = f"百度翻译API错误: {result.get('error_msg', '未知错误')}"
                logger.error(error_msg)
                translated_text = error_msg
                confidence = 0.0
            else:
                translated_text = result["trans_result"][0]["dst"]
                confidence = min(95.0, max(60.0, 100 - len(text) * 0.1))

            end_time = datetime.now()
            translation_time = (end_time - start_time).total_seconds()

            return TranslationResult(
                source_text=text,
                source_language=code_to_name.get(from_lang, from_lang),
                target_language=code_to_name.get(to_lang, to_lang),
                translated_text=translated_text,
                confidence=confidence,
                translation_time=translation_time,
                model_used="baidu-translate",
                additional_info=result
            )

        except Exception as e:
            logger.error(f"云翻译请求失败: {str(e)}")
            return TranslationResult(
                source_text=text,
                source_language=source_lang,
                target_language=target_lang,
                translated_text=f"翻译请求失败: {str(e)}",
                confidence=0.0,
                translation_time=0.0,
                model_used="baidu-translate"
            )

    def supported_languages(self) -> List[str]:
        """获取百度API支持的语言列表"""
        return list(self.language_codes.keys())


# === 百度千帆对话模型API ===
class QianfanChatModel:
    """百度千帆对话模型API接口"""

    def __init__(self, api_key: str = None, model_name: str = "ernie-3.5-8k"):
        self.api_key = api_key or os.environ.get("QIANFAN_API_KEY")
        self.model_name = model_name
        self.api_endpoint = "https://qianfan.baidubce.com/v2/chat/completions"  # 修正为平台正确地址
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.proxies = self._get_proxies()  # 获取代理配置

        if not self.api_key:
            logger.warning("未提供百度千帆API凭证，大模型功能可能受限")

    def _get_proxies(self) -> Optional[Dict]:
        """获取代理配置（如需通过代理访问网络）"""
        http_proxy = os.environ.get("HTTP_PROXY", os.environ.get("http_proxy"))
        https_proxy = os.environ.get("HTTPS_PROXY", os.environ.get("https_proxy"))
        if http_proxy or https_proxy:
            return {
                "http": http_proxy,
                "https": https_proxy
            }
        return None

    def generate(self, prompt: str, history: List[Dict] = None) -> str:
        """调用百度千帆对话模型生成回答"""
        if not self.api_key:
            return "未配置百度千帆API凭证，无法使用大模型功能"

        try:
            messages = []
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": prompt})

            response = requests.post(
                self.api_endpoint,
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "max_gen_len": 2048
                },
                headers=self.headers,
                proxies=self.proxies,  # 应用代理
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            # 解析模型返回结果
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                logger.error(f"模型返回异常: {result}")
                return "模型返回结果解析失败"

        except Exception as e:
            logger.error(f"调用百度千帆模型失败: {e}")
            return f"大模型调用失败: {e}"


# === 混合调用的问答系统 ===
class HybridQASystem:
    """混合调用百度翻译API和智能千帆大模型的问答系统"""

    def __init__(self, fanyi_appid: str = None, fanyi_secret: str = None,
                 qianfan_api_key: str = None):
        self.translation_model = TranslationModel(fanyi_appid, fanyi_secret)
        self.qianfan_model = QianfanChatModel(qianfan_api_key)
        self.history = []
        self.max_context_length = 1000
        logger.info("混合问答系统初始化完成（翻译+理解一体化）")

    def extract_translation_history(self) -> str:
        """提取翻译历史作为上下文"""
        history_text = ""
        for item in self.history[-5:]:  # 取最近5条记录
            if item["action"] == "translate":
                src = item["source_text"]
                tgt = item["result"]["translated_text"]
                src_lang = item["result"]["source_language"]
                tgt_lang = item["result"]["target_language"]
                history_text += f"[{src_lang}→{tgt_lang}] {src} → {tgt}\n"
        return history_text

    def translate_and_understand(self, text: str, source_lang: str = "auto", target_lang: str = "en") -> Dict:
        """翻译并理解文本，返回翻译结果和理解摘要"""
        # 1. 先进行翻译
        translation_result = self.translation_model.translate(text, source_lang, target_lang)
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "action": "translate",
            "source_text": text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "result": asdict(translation_result)
        })

        # 2. 构建理解提示词
        context = self.extract_translation_history()
        prompt = f"""
        你是一个专业的翻译助手，擅长语言翻译和语义理解。
        以下是最近的翻译历史：
        {context}

        请对以下翻译结果进行理解和分析：
        原文 ({translation_result.source_language}): {translation_result.source_text}
        译文 ({translation_result.target_language}): {translation_result.translated_text}

        请提供：
        1. 译文的语义解释
        2. 相关领域的背景知识（如果有）
        3. 可能的应用场景
        4. 翻译技巧说明（可选）

        请简明扼要地回答。
        """

        # 3. 调用大模型理解
        understanding = self.qianfan_model.generate(prompt)

        return {
            "translation": translation_result,
            "understanding": understanding
        }

    def answer_question(self, question: str) -> QAResult:
        """回答与翻译或理解相关的问题"""
        context = self.extract_translation_history()

        # 1. 判断是否为简单翻译问题（规则匹配）
        question_lower = question.lower()
        if "为什么" in question_lower or "原因" in question_lower:
            for item in self.history:
                if item["action"] == "translate":
                    src = item["source_text"]
                    tgt = item["result"]["translated_text"]
                    if src.lower() in question_lower or tgt.lower() in question_lower:
                        return QAResult(
                            question=question,
                            answer=f"'{src}'翻译为'{tgt}'是百度翻译API的标准译法",
                            confidence=70.0,
                            source_context=context,
                            model_used="rule"
                        )
            return QAResult(
                question=question,
                answer="翻译结果基于百度翻译API的词典和算法",
                confidence=60.0,
                source_context=context,
                model_used="rule"
            )

        # 2. 其他类型问题调用大模型
        else:
            prompt = f"""
            你是一个专业的翻译助手，擅长语言翻译和语义理解。
            以下是最近的翻译历史：
            {context}

            用户问题：{question}

            请基于翻译历史用**中文**回答问题，如果问题与翻译无关，请提供相关领域的专业中文回答。
            """

            answer = self.qianfan_model.generate(prompt)
            return QAResult(
                question=question,
                answer=answer,
                confidence=80.0,
                source_context=context,
                model_used="qianfan"
            )


# === 一体化翻译助手 ===
class IntegratedTranslationAssistant:
    """整合翻译和理解功能的一体化助手"""

    def __init__(self, fanyi_appid: str = None, fanyi_secret: str = None,
                 qianfan_api_key: str = None):
        self.qa_system = HybridQASystem(fanyi_appid, fanyi_secret, qianfan_api_key)
        logger.info("一体化翻译助手初始化完成（支持翻译+理解+问答）")

    def translate(self, text: str, source_lang: str = "auto", target_lang: str = "en") -> Dict:
        """翻译并理解文本"""
        return self.qa_system.translate_and_understand(text, source_lang, target_lang)

    def ask_question(self, question: str) -> QAResult:
        """回答问题"""
        return self.qa_system.answer_question(question)

    def get_history(self, limit: int = 10) -> List[Dict]:
        """获取历史记录"""
        return self.qa_system.history[-limit:]


# === 交互式命令行界面 ===
def run_interactive_mode(fanyi_appid: str = None, fanyi_secret: str = None,
                         qianfan_api_key: str = None):
    """交互式命令行界面（支持翻译+理解+问答）"""
    assistant = IntegratedTranslationAssistant(fanyi_appid, fanyi_secret, qianfan_api_key)
    print("=== 智能翻译助手（翻译+理解一体化）===")
    print("功能:")
    print("  1. 输入文本直接翻译并获取语义理解")
    print("  2. 输入'?'开头的问题进行问答（可关于翻译或其他领域）")
    print("  3. 输入'h'查看历史记录")
    print("  4. 输入's'修改源语言")
    print("  5. 输入't'修改目标语言")
    print("  6. 输入'q'退出")

    source_lang = "auto"
    target_lang = "en"

    while True:
        try:
            print(f"\n当前: 源语言={source_lang} → 目标语言={target_lang}")
            user_input = input("请输入文本或问题: ").strip()

            if user_input.lower() == 'q':
                break
            elif user_input.lower() == 'h':
                history = assistant.get_history(5)
                if not history:
                    print("暂无历史记录")
                    continue
                print("\n最近操作:")
                for i, item in enumerate(history, 1):
                    if item["action"] == "translate":
                        src = item["source_text"]
                        tgt = item["result"]["translated_text"]
                        print(
                            f"{i}. [{item['source_lang']}→{item['target_lang']}] {src[:30]}{'...' if len(src) > 30 else ''} → {tgt[:30]}{'...' if len(tgt) > 30 else ''}")
            elif user_input.lower() == 's':
                source_lang = input("请输入新的源语言 (默认自动): ").strip() or "auto"
                # 验证语言是否支持
                supported_langs = assistant.qa_system.translation_model.supported_languages()
                if source_lang not in supported_langs and source_lang != "auto":
                    print(f"警告: 源语言 '{source_lang}' 暂不支持，已恢复为自动检测")
                    source_lang = "auto"
                continue
            elif user_input.lower() == 't':
                target_lang = input("请输入新的目标语言 (默认英文): ").strip() or "en"
                # 验证语言是否支持
                supported_langs = assistant.qa_system.translation_model.supported_languages()
                if target_lang not in supported_langs:
                    print(f"警告: 目标语言 '{target_lang}' 暂不支持，已恢复为英文")
                    target_lang = "en"
                continue
            elif not user_input:
                continue
            elif user_input.startswith('?'):
                # 处理问答
                question = user_input[1:].strip()
                if not question:
                    print("请输入具体问题内容")
                    continue
                print("\n思考中...")
                qa_result = assistant.ask_question(question)
                model_tag = "(规则)" if qa_result.model_used == "rule" else "(文心一言)"
                print(f"\n问题: {question}")
                print(f"回答 {model_tag}: {qa_result.answer}")
                print(f"置信度: {qa_result.confidence:.1f}%")
            else:
                # 处理翻译+理解
                if not user_input:
                    print("请输入要翻译的文本")
                    continue
                print("\n处理中...")
                result = assistant.translate(user_input, source_lang, target_lang)
                print(f"\n翻译结果 ({result['translation'].target_language}): {result['translation'].translated_text}")
                print(f"置信度: {result['translation'].confidence:.1f}%")
                print(f"耗时: {result['translation'].translation_time:.2f}秒")
                print("\n语义理解:")
                # 处理大模型返回的过长内容（截断显示前500字）
                understanding = result['understanding']
                if len(understanding) > 500:
                    print(understanding[:500] + "...")
                else:
                    print(understanding)

        except KeyboardInterrupt:
            print("\n操作已取消，按'q'可退出程序")
        except requests.exceptions.ProxyError:
            print("\n错误: 代理连接失败，请检查代理配置是否正确")
        except requests.exceptions.ConnectionError:
            print("\n错误: 网络连接失败，请检查网络是否正常")
        except requests.exceptions.Timeout:
            print("\n错误: 请求超时，请重试或检查网络稳定性")
        except Exception as e:
            print(f"\n发生错误: {str(e)}")
            logger.error(f"交互模式错误: {str(e)}")

    print("\n=== 翻译助手已退出 ===")


# === 主函数 ===
def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="智能翻译助手（翻译+理解一体化）")
    parser.add_argument("--text", type=str, help="要翻译的文本")
    parser.add_argument("--question", type=str, help="要提问的问题")
    parser.add_argument("--source", type=str, default="auto", help="源语言")
    parser.add_argument("--target", type=str, default="en", help="目标语言")
    parser.add_argument("--fanyi-appid", type=str, help="百度翻译APP ID")
    parser.add_argument("--fanyi-secret", type=str, help="百度翻译Secret Key")
    parser.add_argument("--qianfan-api-key", type=str, help="百度智能千帆API Key")

    args = parser.parse_args()

    assistant = IntegratedTranslationAssistant(
        fanyi_appid=args.fanyi_appid,
        fanyi_secret=args.fanyi_secret,
        qianfan_api_key=args.qianfan_api_key
    )

    if args.question:
        # 直接提问模式
        print(f"\n问题: {args.question}")
        qa_result = assistant.ask_question(args.question)
        print(f"回答: {qa_result.answer}")
        print(f"置信度: {qa_result.confidence:.1f}%")
    elif args.text:
        # 翻译并理解模式
        result = assistant.translate(args.text, args.source, args.target)
        print(f"\n翻译结果 ({result['translation'].target_language}): {result['translation'].translated_text}")
        print(f"置信度: {result['translation'].confidence:.1f}%")
        print("\n语义理解:")
        print(result['understanding'])
    else:
        # 显示帮助并进入交互模式
        parser.print_help()
        run_interactive_mode(
            args.fanyi_appid, args.fanyi_secret,
            args.qianfan_api_key
        )


if __name__ == "__main__":
    main()
