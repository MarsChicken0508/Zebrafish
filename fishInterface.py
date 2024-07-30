import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

def yolo_calculate(image_path, weight_file_path):
    # 假設這是一個 YOLO 計算的函數
    # 這裡放置你的 YOLO 計算邏輯
    result_image_path = image_path  # 這是個假設，實際上應該返回處理過的圖片路徑
    return result_image_path

def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.gif")])
    if file_path:
        file_path_var.set(file_path)

def select_weight_file():
    weight_file_path = filedialog.askopenfilename(filetypes=[("Weight Files", "*.weights")])
    if weight_file_path:
        weight_value_var.set(weight_file_path)

def perform_yolo():
    file_path = file_path_var.get()
    weight_file_path = weight_value_var.get()

    result_image_path = yolo_calculate(file_path, weight_file_path)
    load_image(result_image_path)
    show_zebrafish_window()

def load_image(image_path):
    try:
        img = tk.PhotoImage(file=image_path)
        image_label.config(image=img)
        image_label.image = img
    except Exception as e:
        messagebox.showerror("錯誤", f"無法加載圖片: {e}")

def show_zebrafish_window():
    new_window = tk.Toplevel(root)
    new_window.title("斑馬魚分析")
    new_window.configure(bg='lightblue')
    
    # 設置新視窗的列和行，使其可以動態調整大小
    for i in range(2):
        new_window.grid_columnconfigure(i, weight=1)
    for i in range(3):
        new_window.grid_rowconfigure(i, weight=1)
    
    ttk.Button(new_window, text="斑馬魚運動軌跡影片", command=zebrafish_video).grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

    ttk.Button(new_window, text="正常佔比", command=lambda: zebrafish_behavior("Normal")).grid(row=1, column=0, padx=5, pady=5, sticky="ew")
    ttk.Button(new_window, text="害怕佔比", command=lambda: zebrafish_behavior("Fear")).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
    ttk.Button(new_window, text="焦慮佔比", command=lambda: zebrafish_behavior("Anxiety")).grid(row=2, column=0, padx=5, pady=5, sticky="ew")
    ttk.Button(new_window, text="壓抑佔比", command=lambda: zebrafish_behavior("Depression")).grid(row=2, column=1, padx=5, pady=5, sticky="ew")

def zebrafish_video():
    # Placeholder for zebrafish video tracking logic
    messagebox.showinfo("斑馬魚影片", "斑馬魚運動軌跡計算完成。")

def zebrafish_behavior(behavior_type):
    # Placeholder for zebrafish behavior calculation logic
    messagebox.showinfo("斑馬魚行為", f"斑馬魚行為 ({behavior_type}) 計算完成。")

# 建立主視窗
root = tk.Tk()
root.title("YOLO 和 LSTM 圖形界面")
root.configure(bg='lightblue')  # 設定背景顏色為淺藍色

# 檔案路徑變數
file_path_var = tk.StringVar()
weight_value_var = tk.StringVar()
result_var = tk.StringVar()

# 設置根容器的列和行，使其可以動態調整大小
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)

# 創建一個框架以容納所有的小部件
main_frame = ttk.Frame(root, padding="10")
main_frame.grid(row=0, column=0, sticky="nsew")
main_frame.configure(style="Main.TFrame")

# 設置 main_frame 的列和行，使其可以動態調整大小
for i in range(3):
    main_frame.grid_columnconfigure(i, weight=1)
for i in range(9):
    main_frame.grid_rowconfigure(i, weight=1)

# 設置樣式
style = ttk.Style()
style.configure("TFrame", background='lightblue')
style.configure("TLabel", background='lightblue')
style.configure("TButton", background='lightblue')

# 檔案選擇
ttk.Label(main_frame, text="Select Image/Video:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
ttk.Entry(main_frame, textvariable=file_path_var, width=50).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
ttk.Button(main_frame, text="Browse", command=select_file).grid(row=0, column=2, padx=5, pady=5, sticky="w")

# 權重值選擇
ttk.Label(main_frame, text="Select Weight Value:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
ttk.Entry(main_frame, textvariable=weight_value_var, width=50).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
ttk.Button(main_frame, text="Browse", command=select_weight_file).grid(row=1, column=2, padx=5, pady=5, sticky="w")

# YOLO 計算按鈕
ttk.Button(main_frame, text="YOLO 辨識結果", command=perform_yolo).grid(row=2, column=1, padx=5, pady=5, sticky="ew")

# 圖像顯示
image_label = ttk.Label(main_frame)
image_label.grid(row=4, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")

# 啟動主迴圈
root.mainloop()
