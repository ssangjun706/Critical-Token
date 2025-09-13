import re
import warnings

warnings.filterwarnings("ignore")


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string


def _fix_brackets(string):
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    return string


def fix_all(string):
    string = _fix_brackets(string)
    string = _fix_sqrt(string)
    string = _fix_fracs(string)
    string = _fix_a_slash_b(string)
    return string


def strip_string(string):
    string = str(string).strip()
    string = string.replace("\n", "")
    string = string.rstrip(".")
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    string = _fix_a_slash_b(string)
    return string


def extract_latex_answer(text: str) -> str:
    text = text.replace("\\[", "$$")
    text = text.replace("\\]", "$$")
    text = text.replace("\\(", "$")
    text = text.replace("\\)", "$")

    if "oxed" in text:
        ans = text.split("oxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()

        extracted_math = a
    else:
        special_tag = "§"
        text = text.replace("$$", special_tag)
        text = text.replace("$", special_tag)

        stack = []
        is_open = False
        i = len(text) - 1
        extracted_math = ""

        while i >= 0:
            char = text[i]

            if char == special_tag and not is_open:
                is_open = True
            elif char == special_tag and is_open:
                break
            elif is_open:
                stack.append(char)
            elif (char.isdigit() or char == ")") and not is_open:
                j = i - 1
                while j >= 0 and (text[j].isdigit() or text[j] in "(-+.,^_* "):
                    j -= 1
                extracted_math = text[j + 1 : i + 1].strip()
                break

            i -= 1

        if is_open and len(stack) > 0 and not extracted_math:
            stack = reversed(stack)
            extracted_math = "".join(stack)

        if not extracted_math:
            return ""

    equals_match = re.search(r".*=\s*(.+)$", extracted_math)
    if equals_match:
        extracted_math = equals_match.group(1).strip()

    result = strip_string(extracted_math)
    result = fix_all(result)
    return result
