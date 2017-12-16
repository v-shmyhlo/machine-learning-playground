import numpy as np
import string

n_to_month = {
    1: "jan",
    2: "feb",
    3: "mar",
    4: "apr",
    5: "may",
    6: "jun",
    7: "jul",
    8: "aug",
    9: "sep",
    10: "oct",
    11: "nov",
    12: "dec"
}

special_symbols = {'<p>': 0, '</s>': 1}
vocab = string.ascii_lowercase + '1234567890 /'
sym2id = dict(
    zip(vocab, range(len(special_symbols),
                     len(special_symbols) + len(vocab))))
sym2id = {**special_symbols, **sym2id}
id2sym = {v: k for k, v in sym2id.items()}

pad = sym2id['<p>']
eos = sym2id['</s>']

vocab_size = len(id2sym)
assert vocab_size == len(id2sym) and vocab_size == len(sym2id)

# def text_to_nums(content):
#   result = []
#
#   for ch in content:
#     result.append(char_to_num[ch])
#
#   return result
#
#
# def add_zero_if_needed(el):
#   if (el < 10):
#     return "0" + str(el)
#   else:
#     return str(el)

# def get_el():
#   day = np.random.randint(1, 31)
#   month = np.random.randint(1, 13)
#   year = np.random.randint(0, 100)
#
#   seq = add_zero_if_needed(day) + "/" + add_zero_if_needed(
#       month) + "/" + add_zero_if_needed(year)
#   target = add_zero_if_needed(
#       day) + " " + monthes[month] + " 19" + add_zero_if_needed(year)
#
#   return text_to_nums(seq), text_to_nums(target)

# def nums_to_text(content):
#   result = ''
#   for ch in content:
#     if ch in num_to_char:
#       result += num_to_char[ch]
#     else:
#       result += "?"
#   return result

# def get_batch(batch_size):
#   inputs, targets = [], []
#   for _ in range(batch_size):
#     i, t = get_el()
#     inputs.append(i + [1])
#     targets.append(t + [1])
#
#   inputs = np.reshape(inputs, (batch_size, -1, 1, 1))
#   targets = np.reshape(targets, (batch_size, -1, 1, 1))
#
#   return {"inputs": inputs, "targets": targets, "target_space_id": 2}


def sample_date():
  day = np.random.randint(1, 30)
  month = np.random.randint(1, 12)
  year = np.random.randint(14, 98)
  return day, month, year


def sample():
  day, month, year = sample_date()

  source = '/'.join([str(x) for x in [day, month, year]])
  target = '{} {} 19{}'.format(day, n_to_month[month], year)
  return source, target


def encode(text):
  return [sym2id[sym] for sym in text]


def decode(ids):
  return ''.join(id2sym[id] for id in ids)


def gen():
  while True:
    source, target = sample()
    # source = [sym2id['<start>']] + encode(source) + [sym2id['<end>']]
    # target = [sym2id['<start>']] + encode(target) + [sym2id['<end>']]

    source = encode(source)
    target = encode(target) + [eos]
    yield source, target


def main():
  dataset = gen()
  print(next(dataset))
  print(next(dataset))
  print(next(dataset))


if __name__ == '__main__':
  main()
