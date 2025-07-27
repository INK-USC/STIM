import re

def process_file_path(file_path_ls):
    for file_path in file_path_ls:
        with open(file_path) as f:
            lines = f.readlines()
        for idx, line in enumerate(lines):
            if line == '][\n':
                lines[idx-1] = f'{lines[idx-1][:-1]},\n'
        lines = [lines[i] for i in range(len(lines)) if lines[i] != '][\n']
        content = "".join(lines)
        with open(file_path, 'w') as f1:
            f1.write(content)

def extract_answer(continuation: str):
    """
    This is pre-processing step for this task on the generated continuation from the request.
    """
    # This will grab everything after 'answer is' till the end of the string
    # continuation = cut_at_stop_sequence(continuation, ["Instruction:"])
    pattern = r'answer is\s*(.*)'
    match = re.search(pattern, continuation)
    if match:
        match_answer = match.group(1).strip()
    else:
        print("No match found.")
        return continuation

    if len(match_answer) > 0 and match_answer[-1] in ['.', ',', '-']:
        match_answer = match_answer[:-1]
    return match_answer.replace('<', '')