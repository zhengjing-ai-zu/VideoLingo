import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import itertools
import os,sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_nlp_model import init_nlp
from rich import print

"""
基于逗号和冒号拆分文本，并保存拆分后的结果。它使用 NLP（自然语言处理）模型来确保拆分的合理性，避免在不合适的位置切割句子。
主要逻辑：
    确保右侧短语是完整子句
    排除标点和过短的短语
    使用 NLP 判断句法结构
适用场景：
    自动化文本处理
    语音转录后句子优化
    机器翻译中的断句
"""

def is_valid_phrase(phrase):
    # 🔍 Check for subject and verb
    has_subject = any(token.dep_ in ["nsubj", "nsubjpass"] or token.pos_ == "PRON" for token in phrase)
    has_verb = any((token.pos_ == "VERB" or token.pos_ == 'AUX') for token in phrase)
    return (has_subject and has_verb)

def analyze_comma(start, doc, token):
    left_phrase = doc[max(start, token.i - 9):token.i]
    right_phrase = doc[token.i + 1:min(len(doc), token.i + 10)]
    
    suitable_for_splitting = is_valid_phrase(right_phrase) # and is_valid_phrase(left_phrase) # ! no need to chekc left phrase
    
    # 🚫 Remove punctuation and check word count
    left_words = [t for t in left_phrase if not t.is_punct]
    right_words = list(itertools.takewhile(lambda t: not t.is_punct, right_phrase)) # ! only check the first part of the right phrase
    
    if len(left_words) <= 3 or len(right_words) <= 3:
        suitable_for_splitting = False

    return suitable_for_splitting

def split_by_comma(text, nlp):
    doc = nlp(text)
    sentences = []
    start = 0
    
    for i, token in enumerate(doc):
        if token.text == "," or token.text == "，":
            suitable_for_splitting = analyze_comma(start, doc, token)
            
            if suitable_for_splitting :
                sentences.append(doc[start:token.i].text.strip())
                print(f"[yellow]✂️  Split at comma: {doc[start:token.i][-4:]},| {doc[token.i + 1:][:4]}[/yellow]")
                start = token.i + 1
    
    for i, token in enumerate(doc):
        if token.text == ":": # Split at colon
            sentences.append(doc[start:token.i].text.strip())
            print(f"[yellow]✂️  Split at colon: {doc[start:token.i][-4:]}:| {doc[token.i + 1:][:4]}[/yellow]")
                
    
    sentences.append(doc[start:].text.strip())
    return sentences

def split_by_comma_main(nlp):

    with open("output/log/sentence_by_mark.txt", "r", encoding="utf-8") as input_file:
        sentences = input_file.readlines()

    all_split_sentences = []
    for sentence in sentences:
        split_sentences = split_by_comma(sentence.strip(), nlp)
        all_split_sentences.extend(split_sentences)

    with open("output/log/sentence_by_comma.txt", "w", encoding="utf-8") as output_file:
        for sentence in all_split_sentences:
            output_file.write(sentence + "\n")
    
    # delete the original file
    os.remove("output/log/sentence_by_mark.txt")
    
    print("[green]💾 Sentences split by commas saved to →  `sentences_by_comma.txt`[/green]")

if __name__ == "__main__":
    nlp = init_nlp()
    split_by_comma_main(nlp)
    # nlp = init_nlp()
    # test = "So in the same frame, right there, almost in the exact same spot on the ice, Brown has committed himself, whereas McDavid has not."
    # print(split_by_comma(test, nlp))