 #!/usr/Applications/Python python3  
import os
query=''
for line in open("./input.txt"):
    print('line',line)
    line.rstrip()
    line=line.replace(' ','%20')
    if line.endswith('\n'):
        line=line[:-2]
    query+=line
# print(query)
list_url=['https://www.google.com/search?q={}',
          'https://so.csdn.net/so/search?q={}',
          'https://www.zhihu.com/search?q={}',
          'http://www.baidu.com/s?wd={}',
          'https://zzk.cnblogs.com/s?w={}',]
for i in range(5):
    search_url=list_url[i].format(query)
    command_code='open '+'\''+search_url+'\''
    # print(command_code)
    os.system(command_code)
    print('Finished! ',search_url,'~')
