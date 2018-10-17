import nltk
grammar = r"""
NP: {<DT|JJ|NN.*>+}
PP: {<IN><NP>}
VP: {<VB.*><NP|PP|CLAUSE>+$}
CLAUSE: {<NP><VP>}
"""
cp = nltk.RegexpParser(grammar, loop=2)

sentence = [("Mary", "NN"), ("saw", "VBD"), ("the", "DT"), ("cat", "NN"),("sit", "VB"), ("on", "IN"), ("the", "DT"), ("mat", "NN")]

tree = cp.parse(sentence)

print(tree)
tree.draw()
