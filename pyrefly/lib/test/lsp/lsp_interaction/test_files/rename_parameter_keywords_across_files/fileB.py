from fileA import test

def unrelated(param):
    return param

result = test(param="hello")
other = unrelated(param="skip")
