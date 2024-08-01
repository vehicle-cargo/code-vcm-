import pandas as pd
from queue import Queue
import ast

# 读取CSV文件
df = pd.read_csv('train/vehicle_dataset_final.csv', header=0)
df.columns = df.columns.str.strip()  # 去除列名的前后空格

# 初始化一个队列
vehicle_queue = Queue()

# 逐行将CSV文件中的数据存入队列
for idx, row in df.iterrows():
    try:
        vehicle_data = [
            row['type'],
            int(row['quality']),
            int(row['volumn']),
            int(row['length']),
            ast.literal_eval(row['startPoint']),
            ast.literal_eval(row['endPoint']),
            row['StartProvince'],
            row['endProvince'],
            ast.literal_eval(row['time']),
            int(row['i_index'])
        ]
        vehicle_queue.put(vehicle_data)
    except ValueError as e:
        print(f"Error processing row {idx + 1}: {e}")
    except SyntaxError as e:
        print(f"Error parsing list in row {idx + 1}: {e}")

# 创建最终的队列字典
final_vehicle_queue = {}
idx = 1
while not vehicle_queue.empty():
    final_vehicle_queue[idx] = vehicle_queue.get()
    idx += 1

# 打印最终的队列字典
print(f"Total items in the queue: {len(final_vehicle_queue)}")
print(final_vehicle_queue)
