import os

def count_code_lines():
    total_lines = 0
    file_details = []
    
    # 遍历所有Python文件
    for root, _, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = len(f.readlines())
                    total_lines += lines
                    file_details.append((file_path, lines))
                except Exception as e:
                    print(f"无法读取文件 {file_path}: {e}")
    
    # 按文件大小排序并显示
    file_details.sort(key=lambda x: x[1], reverse=True)
    
    print("文件代码行数统计：")
    for file_path, lines in file_details:
        print(f"{file_path}: {lines} 行")
    
    print(f"\n总计代码行数：{total_lines} 行")

if __name__ == "__main__":
    count_code_lines()