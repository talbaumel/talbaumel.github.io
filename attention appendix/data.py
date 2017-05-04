from random import choice, randrange

EOS = "<EOS>"

characters = list("abcd")
characters.append(EOS)

int2char = list(characters)
char2int = {c:i for i,c in enumerate(characters)}

VOCAB_SIZE = len(characters)


def sample_model(min_length, max_lenth):
    random_length = randrange(min_length, max_lenth)                             # Pick a random length
    random_char_list = [choice(characters[:-1]) for _ in range(random_length)]  # Pick random chars
    random_string = ''.join(random_char_list)
    return random_string, random_string[::-1]  # Return the random string and its reverse

MAX_STRING_LEN = 15

train_set = [sample_model(1, MAX_STRING_LEN) for _ in range(3000)]
val_set = [sample_model(1, MAX_STRING_LEN) for _ in range(50)]