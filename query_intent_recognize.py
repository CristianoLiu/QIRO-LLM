import os
import json
from openai import OpenAI
from tqdm import tqdm

os.environ["OPENAI_API_KEY"] = "sk-324ca55a29ae451ead71f468d48b3270"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def openai_intent_only(doc_texts):

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )

    intent_templates = [
        {"标签": "病情诊断", "定义": "已知症状，判断可能的原因"},
        {"标签": "病因分析", "定义": "已知疾病，解释疾病发生的原因"},
        {"标签": "治疗方案", "定义": "已知疾病/症状，给出治疗或缓解的方案"},
        {"标签": "就医建议", "定义": "已知症状/疾病，给出就医建议"},
        {"标签": "指标解读", "定义": "检查结果的数值范围解读"},
        {"标签": "疾病表述", "定义": "疾病属性、症状、表现等相关表述"},
        {"标签": "后果表述", "定义": "疾病/症状/药品等的危害或治疗效果"},
        {"标签": "注意事项", "定义": "病人注意事项及食物影响分析"},
        {"标签": "功效作用", "定义": "食品/药物的功效/作用/副作用"},
        {"标签": "医疗费用", "定义": "疾病/手术/药品/检查的费用"}
    ]

    all_intents = [item["标签"] for item in intent_templates]
    # 构建包含标签和定义的说明文本
    intent_descriptions = "\n".join([f"- {item['标签']}: {item['定义']}" for item in intent_templates])

    prompt = f"""
    请根据给定的查询内容，从预定义的意图类型集合中判断对应的意图标签。请注意，一个查询可能同时涉及多个意图，因此请进行全面分析并列出所有相关的意图。
    可用的意图类型及其定义：
    {intent_descriptions}

    请仔细审查查询的上下文和语义，确保识别出所有适用的意图类型。你的回复必须严格遵循以下 JSON 格式，且不得包含任何额外内容或解释：
    {{"intent_type": "[识别出的意图标签，如有多个请用英文逗号分隔]"}}
    请确保输出格式正确且完整，不包含任何多余的文本或标记。
    用户输入：{doc_texts}
    """

    try:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system",
                 "content": "你是一名专业的医疗意图理解专家，擅长从患者、医生或其他用户的自然语言查询中准确识别与医疗相关的意图。"},
                {"role": "user", "content": prompt}
            ]
        )

        intent_text = response.choices[0].message.content.strip()
        intents = []

        # 优化JSON解析逻辑
        try:
            # 先去除可能的多余字符（比如前后的反引号、空格）
            clean_intent_text = intent_text.strip().strip("`").strip()
            intent_data = json.loads(clean_intent_text)
            intent_str = intent_data.get("intent_type", "")
            # 分割并清洗
            intents = [i.strip() for i in intent_str.split(",") if i.strip()]
        except json.JSONDecodeError:
            # JSON解析失败时，直接处理原始文本（去除特殊字符）
            clean_text = intent_text.strip().strip('"').strip("'").strip()
            intents = [i.strip() for i in clean_text.split(",") if i.strip()]

        # 调试：可选，上线后可删除
        # print(f"调试：intents = {intents}")

        # 严格匹配，确保意图在预定义列表中
        valid_intents = [i for i in intents if i in all_intents]

        return ",".join(valid_intents) if valid_intents else "其他"

    except Exception as e:
        print(f"API调用错误: {e}")
        return "其他"


def process_file(input_filename, output_filename):
    try:
        processed_ids = set()
        try:
            with open(output_filename, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    processed_ids.add(item.get("id"))
        except FileNotFoundError:
            pass

        with open(input_filename, "r", encoding="utf-8") as infile:
            lines = infile.readlines()

        for line in tqdm(lines, desc="Processing", unit="item"):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            qid = item.get("id")
            if qid in processed_ids:
                continue

            text = item.get("text", "")
            intent = openai_intent_only(text)

            result = {
                "id": qid,
                "text": text,
                "intent": intent
            }

            with open(output_filename, "a", encoding="utf-8") as outfile:
                outfile.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"处理完成，结果保存为: {output_filename}")

    except Exception as e:
        print(f"文件处理出错: {e}")


if __name__ == "__main__":
    input_filename = "./data/MedicalRetrieval/queries.jsonl"
    output_filename = "./data/MedicalRetrieval/queries_intent.jsonl"
    process_file(input_filename, output_filename)