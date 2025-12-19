num = int(input().strip())
R_number = []

trans ={
    'A':'2',
    'B':'2',
    'C':'2',
    'D':'3',
    'E':'3',
    'F':'3',
    'G':'4',
    'H':'4',
    'I':'4',
    'J':'5',
    'K':'5',
    'L':'5',
    'M':'6',
    'N':'6',
    'O':'6',
    'P':'7',
    'S':'7',
    'R':'7',
    'T':'8',
    'U':'8',
    'V':'8',
    'W':'9',
    'X':'9',
    'Y':'9'
}

for _ in range(num):
    number = input()
    correct_num = number.translate(str.maketrans(trans))
    std_num = correct_num.replace('-','')
    R_number.append(std_num)

R_number.sort()
count = {}
p_obt = 0

for number in R_number:
    if number in count:
        count[number] += 1
    else:
        count[number] = 1

for number,cnt in count.items():
    if cnt>1:
        format_r = f"{number[:3]}-{number[3:]}"
        print(f'{format_r} {cnt}')
        p_obt += 1

if p_obt ==0:
    print('No duplicates.')

