
'''
사용자로 부터 문자열을 입력받고, 같은 문자가 연속 등장하는 가장 긴 시퀀스를 찾아내어라
그 문자의 개수를 기반으로 피라미드를 출력해라.
'''


str = input("Enter a string:")

def find_longest_char(s: str):
    cur_chr = ''
    cur_cnt = 0
    max_chr = ''
    max_cnt = 0

    for idx,a in enumerate(str):
        if idx != 0 and str[idx-1] == str[idx]:
            cur_cnt += 1
        else:
            cur_cnt = 1
        cur_chr = str[idx]
        
        if max_cnt <= cur_cnt:
            max_cnt = cur_cnt
            max_chr = cur_chr
        
    return max_chr, max_cnt

# 피라미드 출력
def print_pyramid(max_chr:str, max_cnt):
    for i in range(1, max_cnt + 1): 
        # 1, 2, 3
        # ##c
        # #ccc
        # ccccc
        
        # 1,2
        # #c
        # ccc
        print((max_cnt-i) * " " + max_chr * (2*i -1) )
        

max_chr, max_cnt = find_longest_char(str)
print(f"{max_chr},{max_cnt}")

answer = print_pyramid(max_cnt,max_cnt)
print(answer)