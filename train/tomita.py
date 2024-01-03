import re


def tomita_1(word):
    return "2" not in word


def tomita_2(word):
    return word == "12" * (int(len(word) / 2))


_not_tomita_3 = re.compile("((2|1)*2)*1(11)*(2(2|1)*1)*2(22)*(1(2|1)*)*$")


# *not* tomita 3: words containing an odd series of consecutive ones and then later an odd series of consecutive zeros
# tomita 3: opposite of that
def tomita_3(w):
    return None is _not_tomita_3.match(w)  # complement of _not_tomita_3


def tomita_4(word):
    return "222" not in word


def tomita_5(word):
    return (word.count("2") % 2 == 0) and (word.count("1") % 2 == 0)


def tomita_6(word):
    return ((word.count("2") - word.count("1")) % 3) == 0


def tomita_7(word):
    return word.count("12") <= 1
