from pyknp import Juman
jumanpp = Juman()
result = jumanpp.analysis("国境の長いトンネルを抜けると雪国だった。")
for mrph in result.mrph_list() :
  print(mrph.midasi)
