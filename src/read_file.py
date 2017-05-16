import gzip

with gzip.open('../datasets/train/1_threshold.txt.gz', 'rb') as f:
    file_content = f.readlines()
    # print file_content
    linha_1 = file_content[0]
    lista_linha_1 = linha_1.replace(" ", "").replace("\n", "").split(',')
    lista_linha_1 = map(int, lista_linha_1)
    print lista_linha_1
    print len(lista_linha_1)