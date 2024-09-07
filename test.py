a="asd。"
indices = [index for index, char in enumerate(a) if char in ',，。！？\n']
if len(indices) > 0:
    index = indices[0]
    print(index)

    message = a[index + 1:]
    print(message)