import string

PAD = 0
BOS = 1
EOS = 2
BLK = 3

PAD_CHAR = '<pad>'
BOS_CHAR = '<s>'
EOS_CHAR = '</s>'
BLK_CHAR = ' '

VOCAB = [PAD_CHAR, BOS_CHAR, EOS_CHAR, BLK_CHAR] + list(string.ascii_lowercase) + list(string.digits)