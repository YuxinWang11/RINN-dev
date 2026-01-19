import csv
with open('data/s11.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    header = next(reader)
    print('Header:', header[:10])  # 只打印前10个列名
    row1 = next(reader)
    print('Row 1:', row1[:10])  # 只打印前10个值
    row2 = next(reader)
    print('Row 2:', row2[:10])  # 只打印前10个值