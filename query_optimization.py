import os
import json
import re
from openai import OpenAI
from tqdm import tqdm

os.environ["OPENAI_API_KEY"] = "xxx"
os.environ["OPENAI_BASE_URL"] = "xxx"
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

    task_prompt = {
        "Medical Consultation Retrieval": ("将原始问题重写为一个适合检索相似医疗咨询/问诊案例的查询,使用患者视角表达，不要引入过于专业的医学术语"),
        "Medical Knowledge Retrieval": ("将原始问题重写适合检索权威医学知识（指南、教材、文献）的专业查询,将口语化或非标准表达转换为规范医学术语"),
    }
    task = task_prompt.get(task_type, task_prompt["Medical Consultation Retrieval"])

    intent_defs = []
    for intent in intent_label.split(','):
        intent = intent.strip()
        intent_defs.append(f"{intent}：{intent_templates.get(intent, '其他意图')}")
    intent_def_str = '；'.join(intent_defs) if intent_defs else "其他意图：不属于上述类别的其他意图"

    prompt = f"""
    您的任务是根据指定的任务类型{task}和意图类别定义{intent_templates}优化用户提供的医疗和健康相关查询。
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
        response = client.chat.completions.create(
            model="deepseek-r1",
            messages=[
                {"role": "system",
                 "content": "你是一名医疗咨询检索系统的查询优化助手，擅长根据用户的查询文本和指定意图对查询进行改写优化。"},
                {"role": "user", "content": prompt}
            ]
        )

        response_text = response.choices[0].message.content.strip()

        response_text = re.sub(r'^```json|```$', '', response_text, flags=re.MULTILINE).strip()

        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(0)
        else:
            print(f"未找到有效的JSON结构: {response_text}")
            return None

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

def process_file(input_filename, output_filename):
    try:
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

        with open(input_filename, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()

        for line in tqdm(lines, desc="Processing", unit="item"):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            id_ = item.get('id')
            if id_ in processed_ids:
                continue

            original_text = item.get('text', "")
            intent_label = item.get('intent', "")

            rewritten_text = openai_chat(original_text, intent_label)

            new_item = {
                "id": id_,
                "text": original_text,
                "op_text": rewritten_text if rewritten_text else original_text  
            }

            with open(output_filename, 'a', encoding='utf-8') as outfile:
                outfile.write(json.dumps(new_item, ensure_ascii=False) + '\n')

        print(f"处理完成，结果保存为: {output_filename}")
    except Exception as e:
        print(f"文件处理出错: {e}")

input_filename = './data/xxx/queries_intent.jsonl'  
output_filename = './data/xxx/queries_OP_xxx.jsonl'  
# task_type = "Medical Consultation Retrieval"
task_type = "Medical Knowledge Retrieval"
process_file(input_filename, output_filename)

