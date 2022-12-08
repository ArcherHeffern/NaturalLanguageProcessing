def gen_sentences(path):
    with open(path, encoding="utf8") as file:

        for line in file:
            line = line.removesuffix("\n")
            tokenized_document = []
            if not len(line):
                continue
            token = ""
            is_letter = True
            for char in line:
                #check if space and token has content
                if char == " " and token:
                    tokenized_document.append(token)
                    token = ""
                #check if letter or '
                elif char == "'" or char.lower() != char.upper():
                    if not is_letter and token:
                        tokenized_document.append(token)
                        token = ""
                    is_letter = True
                    token += char
                #check if punctuation
                elif char.upper() == char.lower():
                    if is_letter and token:
                        tokenized_document.append(token)
                        token = ""
                    is_letter = False
                    token += char
            if token:
                tokenized_document.append(token)
            yield tokenized_document


def case_sarcastically(text):
    sarcastic_text = ""
    text = text.lower()
    to_lower = True
    for token in text:
        upper_token = token.upper()
        if upper_token == token:
            sarcastic_text += token
            continue
        elif to_lower:
            sarcastic_text += token
        elif not to_lower:
            sarcastic_text += upper_token
        to_lower = not to_lower
    return sarcastic_text


def prefix(s, n):
    if n > len(s) or n <= 0:
        raise ValueError("n must be greater than or equal to 0 and less than the length of s")
    return s[:n]


def suffix(s, n):
    if n > len(s) or n <= 0:
        raise ValueError("n must be greater than or equal to 0 and less than the length of s")
    return s[len(s) - n:]


def sorted_chars(s):
    return sorted(set(s), key=str.lower)
