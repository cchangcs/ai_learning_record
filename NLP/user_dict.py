import jieba
txt = u'欧阳建国是创新办主任也是欢聚时代公司云计算方面的专家'
# 不使用用户字典分词结果
print(','.join(jieba.cut(txt)))
# 使用用户字典分词结果
jieba.load_userdict('user_dict.txt')
print(','.join(jieba.cut(txt)))
