import syllables
from pypinyin import pinyin, Style
from g2p_en import G2p
from typing import Optional
import re

"""
估算 TTS 生成的音频时长。
用于估算文本朗读时长的 Python 脚本。它通过计算文本的音节数，并根据语言的语速参数进行时间估算。
- 支持多种语言（英语、中文、日语、法语、西班牙语、韩语）
- 自动检测文本语言，对不同语言采用不同的音节计算方式
- 处理混合语言文本，分析文本的语言构成，并分别计算音节数和朗读时长
- 考虑标点符号的停顿时间，包括逗号、句号、空格等
"""
class AdvancedSyllableEstimator:
    def __init__(self):
        """
        self.duration_params：每种语言的平均音节朗读时间（单位：秒）。
        self.lang_patterns：用于匹配不同语言的文本。
        self.lang_joiners：定义不同语言的单词/字符连接方式（例如，中文单字间不加空格）。
        self.punctuation：
            mid：句中停顿符号（如逗号、分号）
            end：句末停顿符号（如句号、感叹号）
            space：空格
            pause：定义停顿时间
        """
        self.g2p_en = G2p()
        self.duration_params = {'en': 0.225, 'zh': 0.21, 'ja': 0.21, 'fr': 0.22, 'es': 0.22, 'ko': 0.21, 'default': 0.22}
        self.lang_patterns = {
            'zh': r'[\u4e00-\u9fff]', 'ja': r'[\u3040-\u309f\u30a0-\u30ff]',
            'fr': r'[àâçéèêëîïôùûüÿœæ]', 'es': r'[áéíóúñ¿¡]', 'en': r'[a-zA-Z]+', 'ko': r'[\uac00-\ud7af\u1100-\u11ff]'}
        self.lang_joiners = {'zh': '', 'ja': '', 'en': ' ', 'fr': ' ', 'es': ' ', 'ko': ' '}
        self.punctuation = {
            'mid': r'[，；：,;、]+', 'end': r'[。！？.!?]+', 'space': r'\s+',
            'pause': {'space': 0.15, 'default': 0.1}
        }

    def estimate_duration(self, text: str, lang: Optional[str] = None) -> float:
        """
        朗读时长估算
        """
        syllable_count = self.count_syllables(text, lang)
        return syllable_count * self.duration_params.get(lang or 'default')

    def count_syllables(self, text: str, lang: Optional[str] = None) -> int:
        """
        音节统计
            英语：使用 syllables 库或 G2p 进行音节估算。
            中文：基于 pypinyin 计算拼音音节数。
            日语：去掉促音 (っ)、长音 (ー)，统计假名与汉字数量。
            法语 & 西班牙语：
            统计元音数
            法语：去掉单词结尾的 e。
            韩语：直接统计韩文音节字符数量。
            默认：按空格分词，计算单词数。
        """
        if not text.strip(): return 0
        lang = lang or self._detect_language(text)
        
        vowels_map = {
            'fr': 'aeiouyàâéèêëîïôùûüÿœæ',
            'es': 'aeiouáéíóúü'
        }
        
        if lang == 'en':
            return self._count_english_syllables(text)
        elif lang == 'zh':
            text = re.sub(r'[^\u4e00-\u9fff]', '', text)
            return len(pinyin(text, style=Style.NORMAL))
        elif lang == 'ja':
            text = re.sub(r'[きぎしじちぢにひびぴみり][ょゅゃ]', 'X', text)
            text = re.sub(r'[っー]', '', text)
            return len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]', text))
        elif lang in ('fr', 'es'):
            text = re.sub(r'e\b', '', text.lower()) if lang == 'fr' else text.lower()
            return max(1, len(re.findall(f'[{vowels_map[lang]}]+', text)))
        elif lang == 'ko':
            return len(re.findall(r'[\uac00-\ud7af]', text))
        return len(text.split())

    def _count_english_syllables(self, text: str) -> int:
        """
        处理英语音节
            优先使用 syllables.estimate(word) 计算音节。
            如果失败，使用 g2p_en（Grapheme-to-Phoneme）转换，将单词转换为音素，然后计算音素中的元音个数。
        """
        total = 0
        for word in text.strip().split():
            try:
                total += syllables.estimate(word)
            except:
                phones = self.g2p_en(word)
                total += max(1, len([p for p in phones if any(c in p for c in 'aeiou')]))
        return max(1, total)

    def _detect_language(self, text: str) -> str:
        """
        语言检测
            遍历语言正则表达式，检测文本是否包含该语言的字符。
            如果匹配到多个语言，以第一个匹配的为准。
            默认返回 en（英语）。
        """
        for lang, pattern in self.lang_patterns.items():
            if re.search(pattern, text): return lang
        return 'en'

    def process_mixed_text(self, text: str) -> dict:
        """
        处理混合语言文本
            按空格和标点分割文本。
            计算不同语言的音节数，并累加朗读时间。
            对空格和标点添加额外停顿时间。
        """
        if not text or not isinstance(text, str):
            return {
                'language_breakdown': {},
                'total_syllables': 0,
                'punctuation': [],
                'spaces': [],
                'estimated_duration': 0
            }
            
        result = {'language_breakdown': {}, 'total_syllables': 0, 'punctuation': [], 'spaces': []}
        segments = re.split(f"({self.punctuation['space']}|{self.punctuation['mid']}|{self.punctuation['end']})", text)
        total_duration = 0
        
        for i, segment in enumerate(segments):
            if not segment: continue
            
            if re.match(self.punctuation['space'], segment):
                prev_lang = self._detect_language(segments[i-1]) if i > 0 else None
                next_lang = self._detect_language(segments[i+1]) if i < len(segments)-1 else None
                if prev_lang and next_lang and (self.lang_joiners[prev_lang] == '' or self.lang_joiners[next_lang] == ''):
                    result['spaces'].append(segment)
                    total_duration += self.punctuation['pause']['space']
            elif re.match(f"{self.punctuation['mid']}|{self.punctuation['end']}", segment):
                result['punctuation'].append(segment)
                total_duration += self.punctuation['pause']['default']
            else:
                lang = self._detect_language(segment)
                if lang:
                    syllables = self.count_syllables(segment, lang)
                    if lang not in result['language_breakdown']:
                        result['language_breakdown'][lang] = {'syllables': 0, 'text': ''}
                    result['language_breakdown'][lang]['syllables'] += syllables
                    result['language_breakdown'][lang]['text'] += (self.lang_joiners[lang] + segment 
                        if result['language_breakdown'][lang]['text'] else segment)
                    result['total_syllables'] += syllables
                    total_duration += syllables * self.duration_params.get(lang, self.duration_params['default'])
        
        result['estimated_duration'] = total_duration
        
        return result
    
def init_estimator():
    return AdvancedSyllableEstimator()

def estimate_duration(text: str, estimator: AdvancedSyllableEstimator):
    if not text or not isinstance(text, str):
        return 0
    return estimator.process_mixed_text(text)['estimated_duration']

# 使用示例
if __name__ == "__main__":
    estimator = init_estimator()
    print(estimate_duration('你好', estimator))

    # 测试用例
    test_cases = [
        # "Hello world this is a test",  # 纯英文
        # "你好世界 这是一个测试",      # 中文带空格
        # "Hello 你好 world 世界",      # 中英混合
        # "The weather is nice 所以我们去公园",  # 中英混合带空格
        # "我们需要在输出中体现空格的停顿时间",
        # "I couldn't help but notice the vibrant colors of the autumn leaves cascading gently from the trees"
        "가을 나뭇잎이 부드럽게 떨어지는 생생한 색깔을 주목하지 않을 수 없었다"
    ]
    
    for text in test_cases:
        result = estimator.process_mixed_text(text)
        print(f"\nText: {text}")
        print(f"Total syllables: {result['total_syllables']}")
        print(f"Estimated duration: {result['estimated_duration']:.2f}s")
        print("Language breakdown:")
        for lang, info in result['language_breakdown'].items():
            print(f"- {lang}: {info['syllables']} syllables ({info['text']})")
        print(f"Punctuation: {result['punctuation']}")
        print(f"Spaces: {result['spaces']}")