import time


class Timer:
    STATE = True

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        if not Timer.STATE:
            return
        self.start_time = time.time()
        return self  # 可以返回 self 以便在 with 块内访问

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not Timer.STATE:
            return
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        print(f">>>>>>代码{self.name} 运行时间: {self.elapsed_time:.4f} 秒")
