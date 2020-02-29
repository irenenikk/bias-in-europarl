def read_file_lines(path):
    file = open(path, 'r')
    return [line.strip() for line in file.readlines()]