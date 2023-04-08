# 0: 우 1: 가
def text2uga(text):
    toBinary = lambda text: ' '.join(format(ord(x), 'b') for x in text)
    binaried_text = toBinary(text)
    result = binaried_text.replace("0", "우").replace("1", "가")
    return result

# 우: 0, 가: 1
def uga2text(uga):
    binaried_text = uga.replace("우", "0").replace("가", "1")
    input_string = [int(binary, 2) for binary in binaried_text.split(" ")]

    toChar = lambda text: "".join(chr(x) for x in text)
    result = toChar(input_string)

    return result


mode = ""
while mode not in ["uga", "kor"]:
    mode = input("모드를 선택해주세요. (uga: 우가어, kor: 한국어): ").lower()

while True:
    input_text = input(">>> ")
    result = uga2text(input_text) if mode == "uga" else text2uga(input_text)

    print(f">>> {result}")