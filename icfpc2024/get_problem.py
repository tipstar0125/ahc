import requests

d = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\]^_`|~ \n"
dd = {}
for i, c in enumerate(d):
    dd[c] = i


def encode(s):
    ret = "S"
    for c in s:
        ret += chr(dd[c] + ord("!"))
    return ret


def decode(s):
    t = ""
    for c in s[1:]:
        n = ord(c) - ord("!")
        if d[n] == " ":
            print(t, end="")
            t = ""
        if d[n] == "\n":
            print(t)
            t = ""
        else:
            t += d[n]

url = "https://boundvariable.space/communicate"
headers = {"Authorization": "Bearer 9984f502-6bcd-4c37-8d85-c672b20fb795"}
data = "get lambdaman"
encode_data = encode(data)
res = requests.post(url=url, headers=headers, data=encode_data).content.decode("utf-8")
# print(res)
decode(res)
