text = input(" ")
binary = lambda text: ' '.join(format(ord(x), 'b') for x in text)
# 0: 전, 1: 현, 2: 준
binaried = binary(text)
print(binaried)

result = ''
for n in list(binaried):
    if n == "0":
        result += '우'
    elif n == "1":
        result += '가'
    else:
        result += " "

print(result[:-1])  # 마지막 '-' 제거