from tqdm import tqdm

tags = ['javascript', 'java', 'python', 'ruby', 'php', 'c++', 'c#', 'go', 'scala', 'swift']

with open('stackoverflow.10kk.tsv') as f, open('stackoverflow.vw', 'w') as to_f:
    for line in tqdm(f):
        splited_line = line.split('\t')
        if len(splited_line) != 2:
            continue

        text, text_tags = splited_line
        text = text.replace('|', ' ').replace(':', ' ').replace('  ', ' ')

        text_tags = set(text_tags.replace('\n', '').split())
        need_tags = set(tags).intersection(text_tags)
        if len(need_tags) != 1:
            continue

        tag_index = tags.index(need_tags.pop())

        to_f.write('%d |text %s\n' % (tag_index, text))
