import csv
import json
import pickle



data = []
ability = {}
dict_data = {}


start = True

with open('data/ability.csv', newline='', encoding='utf-8-sig') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for row in csvreader:
        if start:
            start = False
            continue
        if row[1] in ability:
            ability[row[1]].append(row[0])
        else:
            ability[row[1]] = [row[0]]

start = True

with open('data/skills.csv', newline='', encoding='utf-8-sig') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for row in csvreader:
        if start:
            start = False
            continue
        data.append({'id': row[0].strip(), 'description': row[1].strip(), 'name': row[2].strip(), 
            'q_digit': row[5].strip(), 'class_1': ''.join([i for i in row[6].strip() if not i.isdigit()]) , 'class_2': ''.join([i for i in row[7].strip() if not i.isdigit()]), 'objective': row[9].strip(),
            'abilities': ability[row[0]]})
        dict_data[row[0].strip()] = {'description': row[1].strip(), 'name': row[2].strip(), 
            'q_digit': row[5].strip(), 'class_1': ''.join([i for i in row[6].strip() if not i.isdigit()]) , 'class_2': ''.join([i for i in row[7].strip() if not i.isdigit()]), 'objective': row[9].strip(),
            'abilities': ability[row[0]]}

with open('data/skills.jsonl', 'w', encoding='UTF-8-sig') as outfile:
    for entry in data:
        json.dump(entry, outfile, ensure_ascii=False)
        outfile.write('\n')


with open('data/skills_dict.pkl', 'wb') as f:
    pickle.dump(dict_data, f, pickle.HIGHEST_PROTOCOL)