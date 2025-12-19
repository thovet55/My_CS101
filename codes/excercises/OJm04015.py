mails = []
try:
    while True:
        l = input().strip()
        if l == '':
            break
        mails.append(l)
except EOFError:
    pass

def is_valid_email(s):
    if s.count('@') != 1:
        return False
    if s[0] in ['@', '.'] or s[-1] in ['@', '.']:
        return False
    at_index = s.index('@')
    if '.' not in s[at_index+1:]:
        return False
    if s[at_index-1] == '.' or s[at_index+1] == '.':
        return False
    return True

for mail in mails:
    if is_valid_email(mail):
        print('YES')
    else:
        print('NO')