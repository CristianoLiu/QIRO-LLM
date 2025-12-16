import os
import json
import re
from openai import OpenAI
from tqdm import tqdm

os.environ["OPENAI_API_KEY"] = "sk-324ca55a29ae451ead71f468d48b3270"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
def openai_chat(doc_texts, intent_label):
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )

    intent_templates = {
        "病情诊断": "已知症状，判断可能的原因",
        "病因分析": "已知疾病，解释疾病发生的原因",
        "治疗方案": "已知疾病/症状，给出治疗或缓解的方案",
        "就医建议": "已知症状/疾病，给出就医建议",
        "指标解读": "检查结果的数值范围解读",
        "疾病表述": "疾病属性、症状、表现等相关表述",
        "后果表述": "疾病/症状/药品等的危害或治疗效果",
        "注意事项": "病人注意事项及食物影响分析",
        "功效作用": "食品/药物的功效/作用/副作用",
        "医疗费用": "疾病/手术/药品/检查的费用",
    }

    task  = {
    "Medical Consultation Retrieval": "将原始问题重写为一个适合检索相似医疗咨询/问诊案例的查询,使用患者视角表达，不要引入过于专业的医学术语",
    "Medical Knowledge Retrieval": "将原始问题重写适合检索权威医学知识（指南、教材、文献）的专业查询,将口语化或非标准表达转换为规范医学术语",
     }

    # 解析意图标签，获取对应的定义
    intent_defs = []
    for intent in intent_label.split(','):
        intent = intent.strip()
        intent_defs.append(f"{intent}：{intent_templates.get(intent, '其他意图')}")
    intent_def_str = '；'.join(intent_defs) if intent_defs else "其他意图：不属于上述类别的其他意图"

    prompt = f"""
    您的任务是根据指定的任务类型{task}和意图类别{intent_templates}优化用户提供的医疗和健康相关查询。
    优化后的查询应满足以下要求：
    1. 语言应该清晰、简洁、明确，便于理解。
    2. 应该保留原始查询的核心意图，而不引入过于专门化或模糊的术语。
    3. 结构应完整、自然、流畅，适合后续的医学知识检索。
    输出应该只包含一条完整的优化查询语句。
    请使用以下JSON格式返回结果，不要添加任何额外内容，不要包含代码块标记：
    {{
        "rewritten_query": "[改写后的查询语句]"
    }}

    用户输入文本: {doc_texts}
    用户输入意图: {intent_label}
    """
    try:
        # 调用 OpenAI API
        response = client.chat.completions.create(
            model="deepseek-r1",
            messages=[
                {"role": "system",
                 "content": "你是一名医疗咨询检索系统的查询优化助手，擅长根据用户的查询文本和指定意图对查询进行改写优化。"},
                {"role": "user", "content": prompt}
            ]
        )

        response_text = response.choices[0].message.content.strip()

        # 清理干扰字符
        response_text = re.sub(r'^```json|```$', '', response_text, flags=re.MULTILINE).strip()

        # 提取JSON结构
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(0)
        else:
            print(f"未找到有效的JSON结构: {response_text}")
            return None

        # 解析JSON响应
        try:
            result = json.loads(response_text)
            rewritten_query = result.get("rewritten_query", "").strip()
            return rewritten_query if rewritten_query else None
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}，内容: {response_text}")
            return None

    except Exception as e:
        print(f"API调用错误: {e}")
        return None


# 处理文件并保持JSONL格式，每处理一个就保存一次
def process_file(input_filename, output_filename):
    try:
        # 读取已处理的ID，避免重复处理
        processed_ids = set()
        try:
            with open(output_filename, 'r', encoding='utf-8') as prev_outfile:
                for line in prev_outfile:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        processed_ids.add(item.get('id'))
        except FileNotFoundError:
            pass

        # 读取输入文件所有行
        with open(input_filename, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()

        # 处理每一行并即时保存
        for line in tqdm(lines, desc="Processing", unit="item"):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            id_ = item.get('id')
            if id_ in processed_ids:
                continue

            # 获取原始文本和意图标签
            original_text = item.get('text', "")
            intent_label = item.get('intent', "")

            # 调用模型进行查询优化
            rewritten_text = openai_chat(original_text, intent_label)

            # 构建新的JSON对象（仅保留id、original_query、rewritten_query）
            new_item = {
                "id": id_,
                "original_query": original_text,
                "rewritten_query": rewritten_text if rewritten_text else original_text  # 改写失败则保留原文本
            }

            # 每次处理完一个就立即写入文件（追加模式）
            with open(output_filename, 'a', encoding='utf-8') as outfile:
                outfile.write(json.dumps(new_item, ensure_ascii=False) + '\n')

        print(f"处理完成，结果保存为: {output_filename}")
    except Exception as e:
        print(f"文件处理出错: {e}")


# 示例调用
input_filename = './data/MedicalRetrieval/queries_intent.jsonl'  # 原始JSONL文件
output_filename = './data/MedicalRetrieval/queries_OP_qwen1.jsonl'  # 输出JSONL文件

process_file(input_filename, output_filename)