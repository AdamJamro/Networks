import random
from warnings import warn
import filecmp

FLAG = "01111110"
GEN = "1001"


def minus_gen(x: str):
    res = ""
    for i in range(1, len(GEN)):
        if x[i] == GEN[i]:
            res += "0"
        else:
            res += "1"
    return res


def div_rem(data: str):
    next = len(GEN)
    rem = data[:next]
    while next <= len(data):
        if rem[0] == "1":
            rem = minus_gen(rem)    # reminder has len(GEN) - 1 bits
        else:
            rem = rem[1:]

        if next == len(data):
            break
        rem += data[next]
        next += 1
    return rem


def crc(data: str):
    data += "0" * (len(GEN) - 1)
    return div_rem(data)


def encode(data: str, max_datasize=32):
    frames = []
    i = 0

    while i < len(data):
        cur = data[i: i + max_datasize]
        cur += crc(cur)
        cur_frame = ""
        ones = 0
        for c in cur:
            if ones == 5:
                ones = 0
                cur_frame += '0'
            if c == '1':
                ones += 1
            else:
                ones = 0

            cur_frame += c

        frames.append(FLAG + cur_frame + FLAG)
        i += max_datasize

    return "".join(frames)


def decode(data: str, line: int = -1):
    frames = list(filter(None, data.split(FLAG)))
    frame_counter = 0
    decoded = []
    for frame in frames:
        frame_counter += 1
        current = ""
        i = 0
        ones = 0
        while i < len(frame):
            current += frame[i]
            if frame[i] == "1":
                ones += 1
                if ones == 5:
                    ones = 0
                    i += 1
            else:
                ones = 0
            i += 1

        if div_rem(current) != '0' * (len(GEN) - 1):
            if line != -1:
                warn(f"Line no.{line} in frame no.{frame_counter} containing {frame} omitted.")
            else:
                warn(f"Frame no.{frame_counter} containing {frame} omitted.")

        else:
            current = current[:-(len(GEN) - 1)]
            decoded.append(current)

    return "".join(decoded)


def create_frames(source: str, target: str):
    try:
        with open(source) as src:
            lines = src.read().splitlines()
    except FileNotFoundError:
        print("Source file not found.")
        return

    coded_lines = [encode(line) + '\n' for line in lines]
    with open(target, 'w') as tgt:
        tgt.writelines(coded_lines)


def decode_frames(source: str, target: str):
    try:
        with open(source) as src:
            lines = src.read().splitlines()
    except FileNotFoundError:
        print("Source file not found.")
        return

    coded_lines: str = ""
    for line_index in range(len(lines)):
        coded_lines += decode(lines[line_index], line_index + 1) + '\n'

    with open(target, 'w') as tgt:
        tgt.writelines(coded_lines)


def compare_files(file1: str, file2: str):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        for line1, line2 in zip(f1, f2):
            if line1 != line2:
                print("Failure!")
                return 1
        else:
            print("Success!")
            return 0


def test_decoding():
    create_frames("Z.txt", "W.txt")
    decode_frames("W.txt", "Z_retrieved.txt")
    print("testing without corrupted data:")
    compare_files("Z.txt", "Z_retrieved.txt")

    total_bits = 0
    print("\ntesting with random corruption:")
    with open("W.txt", 'r') as encoded_input:
        encoded_data = encoded_input.read().splitlines()

        for line in encoded_data:
            total_bits += len(line)

        num_of_bits_to_corrupt = random.randrange(1, total_bits // 25)    # 1/50 === 4%

        encoded_data_chr_list: list[chr] = []
        for index in range(len(encoded_data)):
            encoded_data_chr_list.append(list(encoded_data[index]))
        # encoded_data_chr_list = [list(encoded_data_line) for encoded_data_line in encoded_data]

        for _ in range(num_of_bits_to_corrupt):
            line_index = random.randint(0, len(encoded_data_chr_list) - 2)
            index = random.randint(0, len(encoded_data_chr_list[line_index]) - 1)

            if encoded_data_chr_list[line_index][index] == '0':
                encoded_data_chr_list[line_index][index] = '1'

            elif encoded_data_chr_list[line_index][index] == '1':
                encoded_data_chr_list[line_index][index] = '0'

        with open("W_corrupted.txt", 'w') as write_input:
            for line in encoded_data_chr_list:
                write_input.write("".join(line) + '\n')

    decode_frames("W_corrupted.txt", "Z_retrieved.txt")
    compare_files("Z.txt", "Z_retrieved.txt")


# print(x)
# x = encode('1011111000000000000000000000000000000000000000000000000000000111110')

if __name__ == "__main__":
    # create_frames("Z.txt", "W.txt")
    # decode_frames("W.txt", "Z_retrieved.txt")
    # decode_frames("W_corrupted.txt", "Z_retrieved.txt")
    # compare_files("Z.txt", "Z_retrieved.txt")
    test_decoding()
