# byt5-Korean-finetune

[byt5-Korean](https://github.com/everdoubling/byt5-Korean)으로 학습한 언어 모델을 이용하여, 세부 태스크별로 추가 학습(fine-tuning)을 진행하는 코드입니다.
byt5-Korean은 한국어에 특성에 맞게 ByT5의 인코딩 방식을 개선하고 한국어와 영어에 집중하여 사전 학습을 진행한 언어 모델입니다.

## Byte encoding for Korean

ByT5Korean 모델은 한국어 자모별로 하나의 토큰을 할당합니다.

```text
id: token
0: <pad>
1: <eos>
2: <unk>
3~258: utf-8 encoding
259~277: beginning consonants(초성), 19개(ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ)
278~298: middle vowel(중성), 21개(ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ)
299~326: final consonant(종성), 무종성+27개(ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ)
327~384: from <extra_id_0> to <extra_id_57>
```

ByT5KoreanTokenizer.py 파일에 토크나이저가 구현되어 있습니다. 실행 예는 다음과 같습니다.
```python
tokenizer_jamo = ByT5KoreanTokenizer()
print(tokenizer_jamo('가힣abc 안녕하세요')['input_ids'])
# [259, 278, 299, 277, 298, 326, 100, 101, 102, 35, 270, 278, 303, 261, 284, 320, 277, 278, 299, 268, 283, 299, 270, 290, 299, 1]
```

# Results

## nsmc
```shell
python finetune.py nsmc.json
```

## References

[byt5-Korean] https://github.com/everdoubling/byt5-Korean
