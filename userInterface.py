import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

def yolo_calculate(image_path, weight_path):
    # 假設這是一個 YOLO 計算的函數
    # 這裡放置你的 YOLO 計算邏輯
    result_image_path = image_path  # 這是個假設，實際上應該返回處理過的圖片路徑
    return result_image_path

def lstm_calculate(image_path):
    # 假設這是一個 LSTM 計算的函數
    # 這裡放置你的 LSTM 計算邏輯
    return "LSTM Calculation Result"

def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.gif")])
    if file_path:
        file_path_var.set(file_path)

def select_weight():
    weight_path = filedialog.askopenfilename(filetypes=[("Weight Files", "*.weights")])
    if weight_path:
        weight_path_var.set(weight_path)

def perform_yolo():
    file_path = file_path_var.get()
    weight_path = weight_path_var.get()
    if not file_path or not weight_path:
        messagebox.showerror("Error", "Please select both file and weight paths.")
        return

    result_image_path = yolo_calculate(file_path, weight_path)
    load_image(result_image_path)

def perform_lstm():
    file_path = file_path_var.get()
    if not file_path:
        messagebox.showerror("Error", "Please select a file path.")
        return

    result = lstm_calculate(file_path)
    result_var.set(result)

def load_image(image_path):
    try:
        img = tk.PhotoImage(file=image_path)
        image_label.config(image=img)
        image_label.image = img
    except Exception as e:
        messagebox.showerror("Error", f"Cannot load image: {e}")

# 建立主視窗
root = tk.Tk()
root.title("YOLO and LSTM GUI")

# 檔案路徑變數
file_path_var = tk.StringVar()
weight_path_var = tk.StringVar()
result_var = tk.StringVar()

# 設置根容器的列和行，使其可以動態調整大小
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)

# 創建一個框架以容納所有的小部件
main_frame = ttk.Frame(root, padding="10")
main_frame.grid(row=0, column=0, sticky="nsew")

# 設置 main_frame 的列和行，使其可以動態調整大小
for i in range(3):
    main_frame.grid_columnconfigure(i, weight=1)
for i in range(6):
    main_frame.grid_rowconfigure(i, weight=1)

# 檔案選擇
ttk.Label(main_frame, text="Select Image/Video:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
ttk.Entry(main_frame, textvariable=file_path_var, width=50).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
ttk.Button(main_frame, text="Browse", command=select_file).grid(row=0, column=2, padx=5, pady=5, sticky="w")

# 權重檔選擇
ttk.Label(main_frame, text="Select Weight File:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
ttk.Entry(main_frame, textvariable=weight_path_var, width=50).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
ttk.Button(main_frame, text="Browse", command=select_weight).grid(row=1, column=2, padx=5, pady=5, sticky="w")

# YOLO 計算按鈕
ttk.Button(main_frame, text="Perform YOLO Calculation", command=perform_yolo).grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

# LSTM 計算按鈕
ttk.Button(main_frame, text="Perform LSTM Calculation", command=perform_lstm).grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

# 顯示結果
ttk.Label(main_frame, text="LSTM Result:").grid(row=4, column=0, padx=5, pady=5, sticky="e")
ttk.Entry(main_frame, textvariable=result_var, width=50, state="readonly").grid(row=4, column=1, padx=5, pady=5, columnspan=2, sticky="ew")

# 圖像顯示
image_label = ttk.Label(main_frame)
image_label.grid(row=5, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")

# 啟動主迴圈
root.mainloop()
